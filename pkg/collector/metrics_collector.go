package collector

import (
	"context"
	"log"
	"os"
	"sync"
	"time"

	"insight-scope/pkg/client"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	metricsv1beta1 "k8s.io/metrics/pkg/client/clientset/versioned"
)

// MetricsCollector collects node resource metrics and sends to Forecaster
type MetricsCollector struct {
	k8sClient     kubernetes.Interface
	metricsClient *metricsv1beta1.Clientset
	nodeName      string

	forecasterClient *client.ForecasterClient
	collectInterval  time.Duration
	stopChan         chan struct{}
	wg               sync.WaitGroup
}

// NewMetricsCollector creates a new metrics collector
func NewMetricsCollector(kubeconfig string) (*MetricsCollector, error) {
	var config *rest.Config
	var err error

	if kubeconfig != "" {
		config, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
	} else {
		config, err = rest.InClusterConfig()
	}
	if err != nil {
		return nil, err
	}

	k8sClient, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	metricsClient, err := metricsv1beta1.NewForConfig(config)
	if err != nil {
		log.Printf("[MetricsCollector] Warning: metrics client creation failed: %v", err)
		// Continue without metrics client, will use simulated values
	}

	// Get node name from environment (set by Kubernetes)
	nodeName := os.Getenv("NODE_NAME")
	if nodeName == "" {
		nodeName = os.Getenv("HOSTNAME")
	}

	return &MetricsCollector{
		k8sClient:        k8sClient,
		metricsClient:    metricsClient,
		nodeName:         nodeName,
		forecasterClient: client.GetForecasterClient(),
		collectInterval:  1 * time.Minute, // Collect every minute
		stopChan:         make(chan struct{}),
	}, nil
}

// Start begins the metrics collection loop
func (mc *MetricsCollector) Start() {
	mc.wg.Add(1)
	go mc.collectLoop()
	log.Printf("[MetricsCollector] Started collecting metrics for node %s every %v", mc.nodeName, mc.collectInterval)
}

// Stop stops the metrics collector
func (mc *MetricsCollector) Stop() {
	close(mc.stopChan)
	mc.wg.Wait()
	log.Printf("[MetricsCollector] Stopped")
}

// collectLoop periodically collects and sends metrics
func (mc *MetricsCollector) collectLoop() {
	defer mc.wg.Done()

	ticker := time.NewTicker(mc.collectInterval)
	defer ticker.Stop()

	// Initial collection
	mc.collectAndSend()

	for {
		select {
		case <-ticker.C:
			mc.collectAndSend()
		case <-mc.stopChan:
			return
		}
	}
}

// collectAndSend collects metrics and sends to Forecaster
func (mc *MetricsCollector) collectAndSend() {
	if mc.nodeName == "" {
		log.Printf("[MetricsCollector] Node name not set, skipping collection")
		return
	}

	cpuUtil, memUtil, gpuUtil, storageIOUtil := mc.collectNodeMetrics()

	mc.forecasterClient.AddNodeMetrics(
		mc.nodeName,
		cpuUtil,
		memUtil,
		gpuUtil,
		storageIOUtil,
	)
}

// collectNodeMetrics collects resource utilization from the node
func (mc *MetricsCollector) collectNodeMetrics() (cpuUtil, memUtil, gpuUtil, storageIOUtil float64) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// Try to get node metrics from metrics-server
	if mc.metricsClient != nil {
		nodeMetrics, err := mc.metricsClient.MetricsV1beta1().NodeMetricses().Get(ctx, mc.nodeName, metav1.GetOptions{})
		if err == nil {
			// Get node capacity for utilization calculation
			node, err := mc.k8sClient.CoreV1().Nodes().Get(ctx, mc.nodeName, metav1.GetOptions{})
			if err == nil {
				// Calculate CPU utilization
				cpuUsage := nodeMetrics.Usage.Cpu().MilliValue()
				cpuCapacity := node.Status.Capacity.Cpu().MilliValue()
				if cpuCapacity > 0 {
					cpuUtil = float64(cpuUsage) / float64(cpuCapacity)
				}

				// Calculate Memory utilization
				memUsage := nodeMetrics.Usage.Memory().Value()
				memCapacity := node.Status.Capacity.Memory().Value()
				if memCapacity > 0 {
					memUtil = float64(memUsage) / float64(memCapacity)
				}

				log.Printf("[MetricsCollector] Node %s: CPU=%.2f%%, Memory=%.2f%%",
					mc.nodeName, cpuUtil*100, memUtil*100)
			}
		} else {
			log.Printf("[MetricsCollector] Failed to get node metrics: %v, using simulated values", err)
			cpuUtil, memUtil = mc.simulateMetrics()
		}
	} else {
		cpuUtil, memUtil = mc.simulateMetrics()
	}

	// GPU utilization (would need DCGM or nvidia-smi integration)
	gpuUtil = mc.collectGPUMetrics()

	// Storage I/O utilization (would need iostat or similar)
	storageIOUtil = mc.collectStorageIOMetrics()

	return
}

// simulateMetrics returns simulated metrics for testing
func (mc *MetricsCollector) simulateMetrics() (cpuUtil, memUtil float64) {
	// Generate somewhat realistic fluctuating values
	hour := float64(time.Now().Hour())

	// Simulate higher usage during work hours (9-18)
	if hour >= 9 && hour <= 18 {
		cpuUtil = 0.4 + (hour-9)/20.0   // 40-80%
		memUtil = 0.5 + (hour-9)/30.0   // 50-80%
	} else {
		cpuUtil = 0.2 + (hour)/50.0     // 20-50%
		memUtil = 0.3 + (hour)/50.0     // 30-60%
	}

	return
}

// collectGPUMetrics collects GPU utilization
func (mc *MetricsCollector) collectGPUMetrics() float64 {
	// TODO: Integrate with DCGM client or nvidia-smi
	// For now, return simulated value
	return 0.0
}

// collectStorageIOMetrics collects storage I/O utilization
func (mc *MetricsCollector) collectStorageIOMetrics() float64 {
	// TODO: Integrate with iostat or /proc/diskstats
	// For now, return simulated value
	return 0.3
}
