package yolo

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// Model represents a YOLO model
type Model struct {
	session *ort.DynamicAdvancedSession
}

// Input represents the input data for YOLO inference
type Input struct {
	// Height and Width are the dimensions of the input image
	Height int
	// Width is the width of the input image
	Width int
	// Pixels should be preprocessed image data in NCHW format [1, 3, 640, 640]
	Pixels []float32
}

// Output represents the output data from YOLO inference
type Output struct {
	// Logits contains class probabilities for each detection
	Logits []float32
	// Boxes contains bounding box coordinates [x, y, width, height]
	Boxes []float32
}

// New creates a new YOLO model instance
func New(modelPath string) (*Model, error) {
	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer sessionOptions.Destroy()

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"pixel_values"},
		[]string{"logits", "pred_boxes"},
		sessionOptions,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}
	return &Model{session: session}, nil
}

// Run performs inference on the input data
func (m *Model) Run(input *Input) (*Output, error) {
	inputTensor, err := ort.NewTensor(ort.NewShape(1, 3, int64(input.Height), int64(input.Width)), input.Pixels)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputs := make([]ort.Value, 2)

	err = m.session.Run([]ort.Value{inputTensor}, outputs)
	if err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}
	defer outputs[0].Destroy()
	defer outputs[1].Destroy()

	logitsTensor := outputs[0].(*ort.Tensor[float32])
	boxesTensor := outputs[1].(*ort.Tensor[float32])

	return &Output{Logits: logitsTensor.GetData(), Boxes: boxesTensor.GetData()}, nil
}

// Close releases resources
func (m *Model) Close() error {
	if m.session != nil {
		return m.session.Destroy()
	}
	return nil
}
