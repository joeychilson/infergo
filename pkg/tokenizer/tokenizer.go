package tokenizer

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"strings"
)

// Tokenizer represents a BERT tokenizer
type Tokenizer struct {
	Vocab     map[string]int
	IDToToken map[int]string
}

// TokenizerOutput represents the output of the tokenizer
type TokenizerOutput struct {
	InputIds      []int64
	AttentionMask []int64
	Tokens        []string
}

// New creates a new Tokenizer instance
func New(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read tokenizer config: %w", err)
	}

	var config struct {
		Model struct {
			Vocab map[string]int `json:"vocab"`
		} `json:"model"`
	}

	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse tokenizer config: %w", err)
	}

	requiredTokens := []string{"[CLS]", "[SEP]", "[PAD]", "[UNK]"}
	for _, token := range requiredTokens {
		if _, ok := config.Model.Vocab[token]; !ok {
			return nil, fmt.Errorf("required token %s not found in vocab", token)
		}
	}

	idToToken := make(map[int]string, len(config.Model.Vocab))
	for token, id := range config.Model.Vocab {
		idToToken[id] = token
	}

	return &Tokenizer{Vocab: config.Model.Vocab, IDToToken: idToToken}, nil
}

// Encode tokenizes the input text and returns the input IDs and attention mask
func (t *Tokenizer) Encode(text string, maxLength int) (*TokenizerOutput, error) {
	pattern := regexp.MustCompile(`\[[^\[\]]+\]|\w+|[^\w\s]+`)
	tokens := pattern.FindAllString(text, -1)

	wordpieceTokens := []string{}
	for _, token := range tokens {
		if strings.HasPrefix(token, "[") && strings.HasSuffix(token, "]") {
			wordpieceTokens = append(wordpieceTokens, token)
			continue
		}

		token = strings.ToLower(token)

		if _, ok := t.Vocab[token]; ok {
			wordpieceTokens = append(wordpieceTokens, token)
		} else {
			subTokens := t.wordPiece(token)
			wordpieceTokens = append(wordpieceTokens, subTokens...)
		}
	}

	inputIds := []int64{int64(t.Vocab["[CLS]"])}
	finalTokens := []string{"[CLS]"}

	for _, token := range wordpieceTokens {
		if id, ok := t.Vocab[token]; ok {
			inputIds = append(inputIds, int64(id))
			finalTokens = append(finalTokens, token)
		} else {
			inputIds = append(inputIds, int64(t.Vocab["[UNK]"]))
			finalTokens = append(finalTokens, "[UNK]")
		}
	}

	inputIds = append(inputIds, int64(t.Vocab["[SEP]"]))
	finalTokens = append(finalTokens, "[SEP]")

	attentionMask := make([]int64, len(inputIds))
	for i := range attentionMask {
		attentionMask[i] = 1
	}

	if len(inputIds) > maxLength {
		inputIds = inputIds[:maxLength]
		finalTokens = finalTokens[:maxLength]
		attentionMask = attentionMask[:maxLength]
	} else {
		for len(inputIds) < maxLength {
			inputIds = append(inputIds, int64(t.Vocab["[PAD]"]))
			attentionMask = append(attentionMask, 0)
			finalTokens = append(finalTokens, "[PAD]")
		}
	}
	return &TokenizerOutput{
		InputIds:      inputIds,
		AttentionMask: attentionMask,
		Tokens:        finalTokens,
	}, nil
}

// MaskPosition returns the position of the first [MASK] token in the sequence
func (t *Tokenizer) MaskPosition(tokens []string) int {
	for i, token := range tokens {
		if token == "[MASK]" {
			return i
		}
	}
	return -1
}

// VocabSize returns the size of the vocabulary
func (t *Tokenizer) VocabSize() int {
	return len(t.Vocab)
}

func (t *Tokenizer) wordPiece(word string) []string {
	if _, ok := t.Vocab[word]; ok {
		return []string{word}
	}

	tokens := []string{}
	start := 0
	wordLen := len(word)
	for start < wordLen {
		end := wordLen
		var subword string
		found := false

		for end > start {
			substr := word[start:end]
			if start > 0 {
				substr = "##" + substr
			}

			if _, ok := t.Vocab[substr]; ok {
				subword = substr
				found = true
				break
			}
			end--
		}

		if !found {
			return []string{"[UNK]"}
		}

		tokens = append(tokens, subword)
		start = end
	}

	return tokens
}
