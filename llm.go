//go:build js && wasm
// +build js,wasm

package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"syscall/js"
)

type TensorInfo struct {
	DataOffsets []uint64 `json:"data_offsets"`
}

type LlamaLayer struct {
	norm1, norm2           []float32
	wq, wk, wv, wo         []float32
	gate, up, down         []float32
}

var (
	modelWeights []byte
	tensors      map[string]TensorInfo
	
	embed, norm_f, lm_head []float32
	layers                 []LlamaLayer
	kvCache                []float32
	
	vocab    map[string]int
	revVocab map[int]string
	isLoaded bool
)

// --- Math OPs ---

func rmsnorm(out, x, weight []float32) {
	dim := len(x)
	var ss float32
	for i := 0; i < dim; i++ { ss += x[i] * x[i] }
	ss /= float32(dim)
	ss += 1e-5
	ss = float32(1.0 / math.Sqrt(float64(ss)))
	for i := 0; i < dim; i++ { out[i] = weight[i] * (ss * x[i]) }
}

func matmul(out, x, w []float32) {
	d_out := len(out)
	d_in := len(x)
	for i := 0; i < d_out; i++ {
		var val float32
		row := w[i*d_in : (i+1)*d_in]
		for j := 0; j < d_in; j++ { val += row[j] * x[j] }
		out[i] = val
	}
}

func applyRoPE(q, k []float32, pos, nHead, nKvHead, headDim int) {
	for i := 0; i < headDim; i += 2 {
		freq := 1.0 / math.Pow(10000.0, float64(i)/float64(headDim))
		val := float64(pos) * freq
		fcr, fci := float32(math.Cos(val)), float32(math.Sin(val))

		for h := 0; h < nHead; h++ {
			q0, q1 := q[h*headDim+i], q[h*headDim+i+1]
			q[h*headDim+i] = q0*fcr - q1*fci
			q[h*headDim+i+1] = q0*fci + q1*fcr
		}
		for h := 0; h < nKvHead; h++ {
			k0, k1 := k[h*headDim+i], k[h*headDim+i+1]
			k[h*headDim+i] = k0*fcr - k1*fci
			k[h*headDim+i+1] = k0*fci + k1*fcr
		}
	}
}

func swiglu(out, gate, up []float32) {
	for i := 0; i < len(out); i++ {
		x := gate[i]
		silu := x / (1.0 + float32(math.Exp(float64(-x))))
		out[i] = silu * up[i]
	}
}

// --- Forward Pass ---

func forward(pos, token int) int {
	dim := 576
	headDim := 64
	nHead := 9
	nKvHead := 3

	// Safety check to prevent context overflow
	if pos >= 2048 {
		pos = 2047 
	}

	x := make([]float32, dim)
	// Safety bound for vocab
	if token*dim >= len(embed) { token = 0 }
	copy(x, embed[token*dim:(token+1)*dim])

	for l := 0; l < 30; l++ {
		xb := make([]float32, dim)
		rmsnorm(xb, x, layers[l].norm1)

		q, k, v := make([]float32, dim), make([]float32, 192), make([]float32, 192)
		matmul(q, xb, layers[l].wq)
		matmul(k, xb, layers[l].wk)
		matmul(v, xb, layers[l].wv)

		applyRoPE(q, k, pos, nHead, nKvHead, headDim)

		xb2 := make([]float32, dim)
		for h := 0; h < nHead; h++ {
			kv_h := h / 3
			// Cache Keys and Values
			kIdx := l*786432 + 0*393216 + pos*192 + kv_h*64
			vIdx := l*786432 + 1*393216 + pos*192 + kv_h*64
			copy(kvCache[kIdx:kIdx+64], k[kv_h*64:(kv_h+1)*64])
			copy(kvCache[vIdx:vIdx+64], v[kv_h*64:(kv_h+1)*64])

			scores := make([]float32, pos+1)
			scale := float32(1.0 / math.Sqrt(64.0))
			qh := q[h*64 : (h+1)*64]
			for t := 0; t <= pos; t++ {
				var s float32
				kh := kvCache[l*786432 + t*192 + kv_h*64 : l*786432 + t*192 + kv_h*64 + 64]
				for i := 0; i < 64; i++ { s += qh[i] * kh[i] }
				scores[t] = s * scale
			}

			maxS := scores[0]
			for t := 1; t <= pos; t++ { if scores[t] > maxS { maxS = scores[t] } }
			var sumS float32
			for t := 0; t <= pos; t++ {
				scores[t] = float32(math.Exp(float64(scores[t] - maxS)))
				sumS += scores[t]
			}
			for t := 0; t <= pos; t++ { scores[t] /= sumS }

			out_h := xb2[h*64 : (h+1)*64]
			for t := 0; t <= pos; t++ {
				vh := kvCache[l*786432 + 393216 + t*192 + kv_h*64 : l*786432 + 393216 + t*192 + kv_h*64 + 64]
				for i := 0; i < 64; i++ { out_h[i] += scores[t] * vh[i] }
			}
		}

		matmul(xb, xb2, layers[l].wo)
		for i := 0; i < dim; i++ { x[i] += xb[i] }

		rmsnorm(xb, x, layers[l].norm2)
		gate, up := make([]float32, 1536), make([]float32, 1536)
		matmul(gate, xb, layers[l].gate)
		matmul(up, xb, layers[l].up)
		swiglu(gate, gate, up)

		down := make([]float32, dim)
		matmul(down, gate, layers[l].down)
		for i := 0; i < dim; i++ { x[i] += down[i] }
	}

	rmsnorm(x, x, norm_f)
	logits := make([]float32, 49152)
	matmul(logits, x, lm_head)

	best := 0
	maxVal := float32(-1e9)
	for i, val := range logits {
		if val > maxVal {
			maxVal = val
			best = i
		}
	}
	return best
}

// --- Safetensors Loading ---

func safeGetF32(key string) ([]float32, error) {
	t, ok := tensors[key]
	if !ok {
		return nil, fmt.Errorf("Missing tensor: %s", key)
	}
	if len(t.DataOffsets) < 2 {
		return nil, fmt.Errorf("Corrupted offset for: %s", key)
	}
	
	start, end := t.DataOffsets[0], t.DataOffsets[1]
	if start >= uint64(len(modelWeights)) || end > uint64(len(modelWeights)) {
		return nil, fmt.Errorf("OOB offset for: %s", key)
	}

	data := modelWeights[start:end]
	f := make([]float32, len(data)/2)
	
	for i := 0; i < len(f); i++ {
		b0 := uint32(data[i*2])
		b1 := uint32(data[i*2+1])
		u := (b1 << 24) | (b0 << 16)
		f[i] = math.Float32frombits(u)
	}
	return f, nil
}

func initModel(this js.Value, args []js.Value) any {
	jsBuffer, vocabStr := args[0], args[1].String()
	length := jsBuffer.Get("length").Int()

	modelWeights = make([]byte, length)
	js.CopyBytesToGo(modelWeights, jsBuffer)

	headerSize := binary.LittleEndian.Uint64(modelWeights[:8])
	var rawHeader map[string]json.RawMessage
	err := json.Unmarshal(modelWeights[8:8+headerSize], &rawHeader)
	if err != nil {
		return fmt.Sprintf("Header JSON parse error: %v", err)
	}
	
	tensors = make(map[string]TensorInfo)
	for key, msg := range rawHeader {
		if key != "__metadata__" {
			var info TensorInfo
			json.Unmarshal(msg, &info)
			tensors[key] = info
		}
	}

	layers = make([]LlamaLayer, 30)
	for l := 0; l < 30; l++ {
		pfx := fmt.Sprintf("model.layers.%d.", l)
		var err error

		layers[l].norm1, err = safeGetF32(pfx + "input_layernorm.weight")
		if err != nil { return err.Error() }
		
		layers[l].wq, err = safeGetF32(pfx + "self_attn.q_proj.weight")
		if err != nil { return err.Error() }
		
		layers[l].wk, err = safeGetF32(pfx + "self_attn.k_proj.weight")
		if err != nil { return err.Error() }
		
		layers[l].wv, err = safeGetF32(pfx + "self_attn.v_proj.weight")
		if err != nil { return err.Error() }
		
		layers[l].wo, err = safeGetF32(pfx + "self_attn.o_proj.weight")
		if err != nil { return err.Error() }
		
		layers[l].norm2, err = safeGetF32(pfx + "post_attention_layernorm.weight")
		if err != nil { return err.Error() }
		
		layers[l].gate, err = safeGetF32(pfx + "mlp.gate_proj.weight")
		if err != nil { return err.Error() }
		
		layers[l].up, err = safeGetF32(pfx + "mlp.up_proj.weight")
		if err != nil { return err.Error() }
		
		layers[l].down, err = safeGetF32(pfx + "mlp.down_proj.weight")
		if err != nil { return err.Error() }
	}
	
	var errEmbed, errNorm error
	embed, errEmbed = safeGetF32("model.embed_tokens.weight")
	norm_f, errNorm = safeGetF32("model.norm.weight")

	if errEmbed != nil { return errEmbed.Error() }
	if errNorm != nil { return errNorm.Error() }

	// Weight Tying Fallback
	var errHead error
	lm_head, errHead = safeGetF32("lm_head.weight")
	if errHead != nil {
		fmt.Println("Notice: lm_head.weight missing. Using weight tying (fallback to embed_tokens).")
		lm_head = embed
	}

	json.Unmarshal([]byte(vocabStr), &vocab)
	revVocab = make(map[int]string)
	for k, v := range vocab { revVocab[v] = k }

	// FIX: Double the size to properly hold Keys AND Values for all 30 layers
	// 30 layers * 2 (K+V) * 2048 max context * 3 heads * 64 head dim = 23592960
	kvCache = make([]float32, 23592960)
	isLoaded = true

	return "SmolLM2 Engine Ready. 135M params cleanly mounted."
}

// --- Bridging & Generation ---

func generateText(this js.Value, args []js.Value) any {
	prompt := args[0].String()
	
	promiseConstructor := js.Global().Get("Promise")
	return promiseConstructor.New(js.FuncOf(func(this js.Value, args []js.Value) any {
		resolve := args[0]
		go func() {
			if !isLoaded { resolve.Invoke("Error: Model not loaded."); return }

			// Fallback tokenizer matching
			prompt = strings.ReplaceAll(prompt, " ", "Ġ")
			tokens := []int{}
			for len(prompt) > 0 {
				bestLen, bestId := 0, 0
				for word, id := range vocab {
					if strings.HasPrefix(prompt, word) && len(word) > bestLen {
						bestLen = len(word)
						bestId = id
					}
				}
				if bestLen == 0 { bestLen = 1 }
				tokens = append(tokens, bestId)
				prompt = prompt[bestLen:]
			}

			// Prefill context
			next := 0
			for i, t := range tokens { next = forward(i, t) }

			// Generate next tokens
			for i := len(tokens); i < len(tokens)+20; i++ {
				word := revVocab[next]
				word = strings.ReplaceAll(word, "Ġ", " ")
				word = strings.ReplaceAll(word, "<|im_end|>", "\n")
				
				js.Global().Call("appendToken", word)
				next = forward(i, next)
			}
			resolve.Invoke("")
		}()
		return nil
	}))
}

func main() {
	c := make(chan struct{})
	js.Global().Set("initModel", js.FuncOf(initModel))
	js.Global().Set("generateText", js.FuncOf(generateText))
	<-c
}