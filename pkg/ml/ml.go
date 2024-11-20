package ml

import (
	"math"
	"sort"
)

// Softmax applies the softmax function to a slice of logits
func Softmax(logits []float32) []float32 {
	maxLogit := float32(math.Inf(-1))
	for _, logit := range logits {
		if logit > maxLogit {
			maxLogit = logit
		}
	}

	var sumExp float32
	probs := make([]float32, len(logits))
	for i, logit := range logits {
		exp := float32(math.Exp(float64(logit - maxLogit)))
		probs[i] = exp
		sumExp += exp
	}

	for i := range probs {
		probs[i] /= sumExp
	}
	return probs
}

// TopK returns indices of k largest elements
func TopK(values []float32, k int) []int {
	if k > len(values) {
		k = len(values)
	}

	type indexedValue struct {
		index int
		value float32
	}

	indexed := make([]indexedValue, len(values))
	for i, v := range values {
		indexed[i] = indexedValue{i, v}
	}

	sort.Slice(indexed, func(i, j int) bool {
		return indexed[i].value > indexed[j].value
	})

	indices := make([]int, k)
	for i := 0; i < k; i++ {
		indices[i] = indexed[i].index
	}
	return indices
}

// IoU calculates Intersection over Union between two boxes (x1, y1, x2, y2)
func IoU(a, b [4]float32) float32 {
	intersectX1 := math.Max(float64(a[0]), float64(b[0]))
	intersectY1 := math.Max(float64(a[1]), float64(b[1]))
	intersectX2 := math.Min(float64(a[2]), float64(b[2]))
	intersectY2 := math.Min(float64(a[3]), float64(b[3]))

	intersectArea := math.Max(0, intersectX2-intersectX1) * math.Max(0, intersectY2-intersectY1)

	areaA := float64((a[2] - a[0]) * (a[3] - a[1]))
	areaB := float64((b[2] - b[0]) * (b[3] - b[1]))

	return float32(intersectArea / (areaA + areaB - intersectArea))
}

// Sigmoid applies the sigmoid function element-wise
func Sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

// L2Normalize performs L2 normalization on a vector
func L2Normalize(x []float32) []float32 {
	var sumSquares float32
	for _, v := range x {
		sumSquares += v * v
	}
	norm := float32(math.Sqrt(float64(sumSquares)))

	normalized := make([]float32, len(x))
	for i, v := range x {
		normalized[i] = v / norm
	}
	return normalized
}
