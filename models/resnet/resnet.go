package resnet

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// Model represents a ResNet model
type Model struct {
	session *ort.DynamicAdvancedSession
}

// Input represents the input data for ResNet inference
type Input struct {
	// Pixels should be preprocessed image data in NCHW format [1, 3, 224, 224]
	Pixels []float32
}

// Output represents the output data from ResNet inference
type Output struct {
	// Logits are the raw model outputs before softmax
	Logits []float32
}

// New creates a new ResNet model instance
func New(modelPath string) (*Model, error) {
	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer sessionOptions.Destroy()

	session, err := ort.NewDynamicAdvancedSession(modelPath, []string{"pixel_values"}, []string{"logits"}, sessionOptions)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}
	return &Model{session: session}, nil
}

// Run performs inference on the input data
func (m *Model) Run(input *Input) (*Output, error) {
	inputTensor, err := ort.NewTensor(ort.NewShape(1, 3, 224, 224), input.Pixels)
	if err != nil {
		return nil, fmt.Errorf("failed to create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outputs := make([]ort.Value, 1)

	err = m.session.Run([]ort.Value{inputTensor}, outputs)
	if err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}
	defer outputs[0].Destroy()

	outputTensor := outputs[0].(*ort.Tensor[float32])
	return &Output{Logits: outputTensor.GetData()}, nil
}

// Close releases resources
func (m *Model) Close() error {
	if m.session != nil {
		return m.session.Destroy()
	}
	return nil
}
