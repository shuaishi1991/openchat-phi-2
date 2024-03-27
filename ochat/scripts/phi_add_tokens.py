import argparse

import transformers
import torch


def add_tokens_to_embedding(added_special_tokens, embedding):
    # Mean embedding, shape: [1, dim]
    new_token_embeddings = torch.mean(embedding.to(torch.float32), dim=0, keepdim=True).to(embedding.dtype)
    # Expand to [N, dim]
    new_token_embeddings = new_token_embeddings.expand(len(added_special_tokens), -1)

    return torch.cat([embedding, new_token_embeddings], dim=0)


def phi_add_tokens(model_path, output_dir, added_special_tokens):
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, legacy=False)
    model = transformers.PhiForCausalLM.from_pretrained(model_path,
                                                            low_cpu_mem_usage=True,
                                                            torch_dtype=torch.bfloat16)
    # Add tokens (tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": added_special_tokens})

    # Add tokens (embedding)
    assert model.model.embed_tokens.weight.requires_grad
    assert model.lm_head.weight.requires_grad
    assert model.lm_head.bias.requires_grad

    model.model.embed_tokens.weight = torch.nn.Parameter(add_tokens_to_embedding(added_special_tokens, model.model.embed_tokens.weight), requires_grad=True)
    model.lm_head.weight            = torch.nn.Parameter(add_tokens_to_embedding(added_special_tokens, model.lm_head.weight), requires_grad=True)
    model.lm_head.bias              = torch.nn.Parameter(add_tokens_to_embedding(added_special_tokens, model.lm_head.bias.unsqueeze(-1)).squeeze(-1), requires_grad=True)

    model.config.vocab_size += len(added_special_tokens)

    # Fix model config (actual token length is 8192)
    print('111111111 model.config.max_position_embeddings: ', model.config.max_position_embeddings)
    assert model.config.max_position_embeddings == 2048
    model.config.max_position_embeddings = 2048

    # Save
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        help="Location of Phi model, or HuggingFace repo ID",
    )
    parser.add_argument(
        "--output-dir",
        help="Location to write resulting model and tokenizer",
    )
    parser.add_argument(
        "--added-special-tokens",
        type=str,
        nargs="+",
        help="Special token list to add"
    )

    phi_add_tokens(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
