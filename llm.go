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

	// FIX: Handle Weight Tying
	// SmolLM saves space by reusing the embedding weights for the lm_head!
	lm_head, err = safeGetF32("lm_head.weight")
	if err != nil {
		fmt.Println("Notice: lm_head.weight missing. Using weight tying (fallback to embed_tokens).")
		lm_head = embed
	}

	json.Unmarshal([]byte(vocabStr), &vocab)
	revVocab = make(map[int]string)
	for k, v := range vocab { revVocab[v] = k }

	kvCache = make([]float32, 11796480)
	isLoaded = true

	return "SmolLM2 Engine Ready. 135M params cleanly mounted."
}