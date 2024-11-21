// pkg/models/bert/bert.go
package bert

import (
	"fmt"

	ort "github.com/yalue/onnxruntime_go"
)

// Model represents a BERT model
type Model struct {
	session *ort.DynamicAdvancedSession
}

// Input represents the input data for BERT inference
type Input struct {
	InputIds      []int64
	AttentionMask []int64
}

// Output represents the output data from BERT inference
type Output struct {
	Logits []float32
}

// New creates a new BERT model instance
func New(modelPath string) (*Model, error) {
	sessionOptions, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("failed to create session options: %w", err)
	}
	defer sessionOptions.Destroy()

	session, err := ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{"input_ids", "attention_mask"},
		[]string{"logits"},
		sessionOptions,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return &Model{session: session}, nil
}

// Run performs inference on the input data
func (m *Model) Run(input *Input) (*Output, error) {
	inputIdsShape := ort.NewShape(1, int64(len(input.InputIds)))
	inputIdsTensor, err := ort.NewTensor(inputIdsShape, input.InputIds)
	if err != nil {
		return nil, fmt.Errorf("failed to create input_ids tensor: %w", err)
	}
	defer inputIdsTensor.Destroy()

	attentionMaskShape := ort.NewShape(1, int64(len(input.AttentionMask)))
	attentionMaskTensor, err := ort.NewTensor(attentionMaskShape, input.AttentionMask)
	if err != nil {
		return nil, fmt.Errorf("failed to create attention_mask tensor: %w", err)
	}
	defer attentionMaskTensor.Destroy()

	inputs := []ort.Value{inputIdsTensor, attentionMaskTensor}
	outputs := make([]ort.Value, 1)

	err = m.session.Run(inputs, outputs)
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
