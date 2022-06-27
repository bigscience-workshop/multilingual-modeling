from transformers import AutoModel

model_0 = AutoModel.from_pretrained()
model_F = AutoModel.from_pretrained()


for (name_0, param_0), (name_F, param_F) in zip(model_0.named_parameters(), model_F.named_parameters()):

	param_name = 
	if "bias" in name_0:
		if "query_key_value":

			# Query, Key, Value are merged in one MLP,
			# so we need to seperate the bias terms
			
			head_size = model_0.config.hidden_size // model_0.config.num_attention_heads
			
			_q_change = None
			_k_change = None
			_v_change = None
			for qkv_bias in [param_0, param_F]:
				qkv_bias = qkv_bias.view(num_attention_heads, 3*head_size)

				if _q_change is None:
					_q_change = qkv_bias[..., :head_size]
				else:
					_q_change -= qkv_bias[..., :head_size]
					_q_change = torch.norm(_q_change)

				if _k_change is None:
					_k_change = qkv_bias[..., head_size: 2 * head_size]
				else:
					_k_change -= qkv_bias[..., head_size: 2 * head_size]
					_k_change = torch.norm(_k_change)

				if _v_change is None:
					_v_change = qkv_bias[..., 2 * head_size:]
				else:
					_v_change -= qkv_bias[..., 2 * head_size:]
					_v_change = torch.norm(_v_change)
		else:
			bias_change = torch.norm(param_0 - param_F)

transformer.h.0.input_layernorm.bias
transformer.h.0.self_attention.query_key_value.bias
transformer.h.0.self_attention.dense.bias
transformer.h.0.post_attention_layernorm.bias
transformer.h.0.mlp.dense_h_to_4h.bias
transformer.h.0.mlp.dense_4h_to_h.bias