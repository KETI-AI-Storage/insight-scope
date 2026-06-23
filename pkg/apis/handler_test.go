package apis

import (
	"strings"
	"testing"
)

// TestCalculateStorageSize locks the storage-size formatting. The original code used
// string(rune(n+'0')) which only produces a correct digit for single-digit n and emits
// a garbage Unicode code point for any value > 9.
func TestCalculateStorageSize(t *testing.T) {
	h := &Handler{}

	cases := []struct {
		name       string
		dataSizeGB int
		want       string
	}{
		{"single digit Gi", 4, "6Gi"},           // 4 * 1.5 = 6
		{"multi digit Gi", 100, "150Gi"},        // 100 * 1.5 = 150
		{"multi digit Ti", 10000, "14Ti"},       // 10000 * 1.5 = 15000 -> 15000/1024 = 14
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := h.calculateStorageSize(tc.dataSizeGB)
			if got != tc.want {
				t.Fatalf("calculateStorageSize(%d) = %q, want %q", tc.dataSizeGB, got, tc.want)
			}
		})
	}
}

// TestGenerateRecommendationGPUReasoning locks the GPU-count interpolation in the
// reasoning string (same rune-conversion bug as above).
func TestGenerateRecommendationGPUReasoning(t *testing.T) {
	h := &Handler{}
	rec := h.generateRecommendation(&RecommendRequest{GPUCount: 12})
	if !strings.Contains(rec.Reasoning, "12 GPUs") {
		t.Fatalf("reasoning = %q, want it to mention %q", rec.Reasoning, "12 GPUs")
	}
}
