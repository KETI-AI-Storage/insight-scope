package client

import (
	"context"
	"fmt"
	"log"
	"os"
	"sync"
	"time"

	pb "insight-scope/proto/forecasterpb"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/types/known/timestamppb"
)

// ForecasterClient is a gRPC client for Node Resource Forecaster
type ForecasterClient struct {
	endpoint  string
	conn      *grpc.ClientConn
	client    pb.NodeResourceForecasterServiceClient
	connected bool
	mu        sync.RWMutex

	// Batch history data before sending
	historyBuffer   map[string][]ResourceSnapshot
	bufferMu        sync.Mutex
	bufferSize      int
	flushInterval   time.Duration
	stopChan        chan struct{}
}

// ResourceSnapshot represents a single resource utilization snapshot
type ResourceSnapshot struct {
	Timestamp         time.Time
	CPUUtilization    float64
	MemoryUtilization float64
	GPUUtilization    float64
	StorageIOUtil     float64
}

var (
	globalForecasterClient *ForecasterClient
	forecasterClientOnce   sync.Once
)

// GetForecasterClient returns the singleton Forecaster client
func GetForecasterClient() *ForecasterClient {
	forecasterClientOnce.Do(func() {
		endpoint := os.Getenv("FORECASTER_ENDPOINT")
		if endpoint == "" {
			endpoint = "node-resource-forecaster.apollo.svc.cluster.local:50055"
		}

		globalForecasterClient = &ForecasterClient{
			endpoint:      endpoint,
			historyBuffer: make(map[string][]ResourceSnapshot),
			bufferSize:    60, // Flush every 60 snapshots (1 hour if 1/min)
			flushInterval: 5 * time.Minute,
			stopChan:      make(chan struct{}),
		}

		// Start background flush goroutine
		go globalForecasterClient.backgroundFlush()

		// Try to connect (non-blocking)
		go func() {
			if err := globalForecasterClient.Connect(); err != nil {
				log.Printf("[ForecasterClient] Initial connection failed: %v", err)
			}
		}()
	})
	return globalForecasterClient
}

// Connect establishes connection to Forecaster
func (c *ForecasterClient) Connect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.connected {
		return nil
	}

	log.Printf("[ForecasterClient] Connecting to %s", c.endpoint)

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(
		ctx,
		c.endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return fmt.Errorf("failed to connect to Forecaster: %w", err)
	}

	c.conn = conn
	c.client = pb.NewNodeResourceForecasterServiceClient(conn)
	c.connected = true
	log.Printf("[ForecasterClient] Connected to Forecaster")
	return nil
}

// Close closes the connection
func (c *ForecasterClient) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	close(c.stopChan)

	if c.conn != nil {
		c.connected = false
		return c.conn.Close()
	}
	return nil
}

// IsConnected returns connection status
func (c *ForecasterClient) IsConnected() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.connected
}

// AddSnapshot adds a resource snapshot to the buffer
func (c *ForecasterClient) AddSnapshot(nodeName string, snapshot ResourceSnapshot) {
	c.bufferMu.Lock()
	defer c.bufferMu.Unlock()

	c.historyBuffer[nodeName] = append(c.historyBuffer[nodeName], snapshot)

	// Check if we should flush
	if len(c.historyBuffer[nodeName]) >= c.bufferSize {
		go c.flushNode(nodeName)
	}
}

// AddNodeMetrics is a convenience method to add metrics
func (c *ForecasterClient) AddNodeMetrics(nodeName string, cpuUtil, memUtil, gpuUtil, storageIOUtil float64) {
	c.AddSnapshot(nodeName, ResourceSnapshot{
		Timestamp:         time.Now(),
		CPUUtilization:    cpuUtil,
		MemoryUtilization: memUtil,
		GPUUtilization:    gpuUtil,
		StorageIOUtil:     storageIOUtil,
	})
}

// flushNode sends buffered data for a specific node
func (c *ForecasterClient) flushNode(nodeName string) {
	c.bufferMu.Lock()
	snapshots := c.historyBuffer[nodeName]
	c.historyBuffer[nodeName] = nil // Clear buffer
	c.bufferMu.Unlock()

	if len(snapshots) == 0 {
		return
	}

	if !c.IsConnected() {
		if err := c.Connect(); err != nil {
			log.Printf("[ForecasterClient] Cannot flush, not connected: %v", err)
			// Put snapshots back in buffer (partial recovery)
			c.bufferMu.Lock()
			c.historyBuffer[nodeName] = append(snapshots, c.historyBuffer[nodeName]...)
			c.bufferMu.Unlock()
			return
		}
	}

	if err := c.submitHistory(nodeName, snapshots); err != nil {
		log.Printf("[ForecasterClient] Failed to submit history for %s: %v", nodeName, err)
	} else {
		log.Printf("[ForecasterClient] Submitted %d snapshots for node %s", len(snapshots), nodeName)
	}
}

// submitHistory sends history data to Forecaster via gRPC
func (c *ForecasterClient) submitHistory(nodeName string, snapshots []ResourceSnapshot) error {
	c.mu.RLock()
	client := c.client
	c.mu.RUnlock()

	if client == nil {
		return fmt.Errorf("client not initialized")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Convert snapshots to proto format
	protoSnapshots := make([]*pb.ResourceSnapshot, len(snapshots))
	for i, s := range snapshots {
		protoSnapshots[i] = &pb.ResourceSnapshot{
			Timestamp:              timestamppb.New(s.Timestamp),
			CpuUtilization:         s.CPUUtilization,
			MemoryUtilization:      s.MemoryUtilization,
			GpuUtilization:         s.GPUUtilization,
			StorageIoUtilization:   s.StorageIOUtil,
		}
	}

	req := &pb.SubmitHistoryRequest{
		NodeName:  nodeName,
		Snapshots: protoSnapshots,
	}

	resp, err := client.SubmitHistoryData(ctx, req)
	if err != nil {
		return fmt.Errorf("gRPC call failed: %w", err)
	}

	if !resp.Accepted {
		return fmt.Errorf("forecaster rejected data")
	}

	log.Printf("[ForecasterClient] Forecaster accepted %d snapshots for %s", resp.SnapshotsProcessed, nodeName)
	return nil
}

// backgroundFlush periodically flushes all buffered data
func (c *ForecasterClient) backgroundFlush() {
	ticker := time.NewTicker(c.flushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ticker.C:
			c.FlushAll()
		case <-c.stopChan:
			return
		}
	}
}

// FlushAll flushes all buffered node data
func (c *ForecasterClient) FlushAll() {
	c.bufferMu.Lock()
	nodes := make([]string, 0, len(c.historyBuffer))
	for nodeName := range c.historyBuffer {
		if len(c.historyBuffer[nodeName]) > 0 {
			nodes = append(nodes, nodeName)
		}
	}
	c.bufferMu.Unlock()

	for _, nodeName := range nodes {
		c.flushNode(nodeName)
	}
}

// GetBufferStats returns current buffer statistics
func (c *ForecasterClient) GetBufferStats() map[string]int {
	c.bufferMu.Lock()
	defer c.bufferMu.Unlock()

	stats := make(map[string]int)
	for nodeName, snapshots := range c.historyBuffer {
		stats[nodeName] = len(snapshots)
	}
	return stats
}

// ForecastNode requests a forecast for a specific node
func (c *ForecasterClient) ForecastNode(nodeName string, horizons []int32) (*pb.ForecastNodeResponse, error) {
	c.mu.RLock()
	client := c.client
	c.mu.RUnlock()

	if client == nil {
		return nil, fmt.Errorf("client not initialized")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req := &pb.ForecastNodeRequest{
		NodeName:                nodeName,
		ForecastHorizonsMinutes: horizons,
	}

	return client.ForecastNodeResources(ctx, req)
}

// GetPeakIdlePrediction requests peak/idle prediction
func (c *ForecasterClient) GetPeakIdlePrediction(nodeName string, lookaheadHours int32) (*pb.PeakIdleResponse, error) {
	c.mu.RLock()
	client := c.client
	c.mu.RUnlock()

	if client == nil {
		return nil, fmt.Errorf("client not initialized")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	req := &pb.PeakIdleRequest{
		NodeName:       nodeName,
		LookaheadHours: lookaheadHours,
	}

	return client.GetPeakIdlePrediction(ctx, req)
}
