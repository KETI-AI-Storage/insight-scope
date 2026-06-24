package client

import (
	"runtime"
	"testing"
	"time"
)

// The forecaster client starts a long-lived backgroundFlush goroutine. Close()
// must actually terminate it (join it), not merely signal it. Otherwise stopping
// the metrics collector leaks the goroutine plus its gRPC connections for the
// remaining process lifetime.
func TestNewForecasterClient_CloseTerminatesBackgroundGoroutine(t *testing.T) {
	base := runtime.NumGoroutine()

	c := newForecasterClient("", "")

	if err := c.Close(); err != nil {
		t.Fatalf("Close returned error: %v", err)
	}

	// Close joins the background goroutine, so the count must be back to baseline.
	if got := runtime.NumGoroutine(); got > base {
		t.Errorf("after Close, goroutines = %d, want <= baseline %d (backgroundFlush leaked)", got, base)
	}
}

// Close must be safe to call more than once (e.g. collector Stop plus any other
// shutdown path on the shared singleton).
func TestForecasterClient_CloseIdempotent(t *testing.T) {
	c := newForecasterClient("", "")

	if err := c.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}

	done := make(chan error, 1)
	go func() { done <- c.Close() }()
	select {
	case err := <-done:
		if err != nil {
			t.Fatalf("second Close: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("second Close hung (not idempotent)")
	}
}
