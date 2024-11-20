package postprocess

import (
	"image"
	"sort"

	"github.com/joeychilson/infergo/pkg/ml"
)

// Classification represents a single class prediction
type Classification struct {
	Label      string
	Class      int
	Confidence float32
}

// ClassificationOptions contains options for processing classification results
type ClassificationOptions struct {
	Labels   map[int]string // Label mapping
	TopK     int            // Number of top predictions to return
	MinScore float32        // Minimum confidence threshold
	Softmax  bool           // Whether to apply softmax to logits
}

// ProcessClassification converts raw logits into structured classifications
func ProcessClassification(logits []float32, opts ClassificationOptions) ([]Classification, error) {
	probabilities := logits
	if opts.Softmax {
		probabilities = ml.Softmax(logits)
	}

	indices := ml.TopK(probabilities, opts.TopK)

	classifications := make([]Classification, 0, len(indices))
	for _, idx := range indices {
		prob := probabilities[idx]
		if prob < opts.MinScore {
			continue
		}

		label, ok := opts.Labels[idx]
		if !ok {
			continue
		}

		classifications = append(classifications, Classification{
			Label:      label,
			Class:      idx,
			Confidence: prob,
		})
	}
	return classifications, nil
}

// DetectionOptions contains options for processing detection results
type DetectionOptions struct {
	Labels        map[int]string // Label mapping
	MaxDetections int            // Maximum number of detections to return
	ConfThreshold float32        // Confidence threshold for detections
	IoUThreshold  float32        // IoU threshold for NMS
}

// Detection represents a detected object with its bounding box
type Detection struct {
	Classification
	Box Box
}

// Box represents a bounding box in pixel coordinates
type Box struct {
	X1 float32
	Y1 float32
	X2 float32
	Y2 float32
}

// ProcessDetections converts raw model outputs into structured detections
func ProcessDetections(logits []float32, boxes []float32, imageSize image.Point, opts DetectionOptions) ([]Detection, error) {
	numBoxes := len(boxes) / 4
	numClasses := len(logits) / numBoxes

	var detections []Detection
	for i := 0; i < numBoxes; i++ {
		boxLogits := logits[i*numClasses : (i+1)*numClasses]
		probs := ml.Softmax(boxLogits)

		var (
			maxProb  float32
			maxClass int
		)
		for c := 0; c < numClasses; c++ {
			if probs[c] > maxProb {
				maxProb = probs[c]
				maxClass = c
			}
		}

		if maxProb < opts.ConfThreshold {
			continue
		}

		label, ok := opts.Labels[maxClass]
		if !ok {
			continue
		}

		box := boxes[i*4 : (i+1)*4]
		detection := Detection{
			Classification: Classification{
				Label:      label,
				Class:      maxClass,
				Confidence: maxProb,
			},
			Box: Box{
				X1: (box[0] - box[2]/2) * float32(imageSize.X),
				Y1: (box[1] - box[3]/2) * float32(imageSize.Y),
				X2: (box[0] + box[2]/2) * float32(imageSize.X),
				Y2: (box[1] + box[3]/2) * float32(imageSize.Y),
			},
		}
		detections = append(detections, detection)
	}
	return NonMaxSuppression(detections, opts.IoUThreshold), nil
}

// NonMaxSuppression applies non-maximum suppression to remove overlapping detections
func NonMaxSuppression(detections []Detection, iouThreshold float32) []Detection {
	if len(detections) == 0 {
		return detections
	}

	sort.Slice(detections, func(i, j int) bool {
		return detections[i].Confidence > detections[j].Confidence
	})

	kept := make([]bool, len(detections))
	var result []Detection

	for i := 0; i < len(detections); i++ {
		if kept[i] {
			continue
		}

		kept[i] = true
		result = append(result, detections[i])

		for j := i + 1; j < len(detections); j++ {
			if kept[j] || detections[i].Class != detections[j].Class {
				continue
			}

			box1 := [4]float32{detections[i].Box.X1, detections[i].Box.Y1, detections[i].Box.X2, detections[i].Box.Y2}
			box2 := [4]float32{detections[j].Box.X1, detections[j].Box.Y1, detections[j].Box.X2, detections[j].Box.Y2}
			if ml.IoU(box1, box2) > iouThreshold {
				kept[j] = true
			}
		}
	}
	return result
}
