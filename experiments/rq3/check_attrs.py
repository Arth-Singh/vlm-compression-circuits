"""Explore model attribute hierarchy for Transformers 5.x compatibility."""
from transformers import LlavaForConditionalGeneration
m = LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf", torch_dtype="auto", device_map="cpu", low_cpu_mem_usage=True
)

print("=== Top-level named children ===")
for name, child in m.named_children():
    print(f"  {name}: {type(child).__name__}")

# Explore model.model (Transformers 5.x path)
if hasattr(m, 'model'):
    print("\n=== m.model named children ===")
    for name, child in m.model.named_children():
        print(f"  {name}: {type(child).__name__}")
        if hasattr(child, '__len__'):
            print(f"    len={len(child)}")

    # Check for layers at various depths
    mm = m.model
    if hasattr(mm, 'layers'):
        print(f"\n  m.model.layers: {type(mm.layers).__name__}, len={len(mm.layers)}")
        if len(mm.layers) > 0:
            layer0 = mm.layers[0]
            print(f"  layer[0] type: {type(layer0).__name__}")
            print(f"  layer[0] children:")
            for n, c in layer0.named_children():
                print(f"    {n}: {type(c).__name__}")

    if hasattr(mm, 'model'):
        print(f"\n  m.model.model exists: {type(mm.model).__name__}")
        if hasattr(mm.model, 'layers'):
            print(f"  m.model.model.layers: len={len(mm.model.layers)}")

    # Check for language_model inside model
    if hasattr(mm, 'language_model'):
        print(f"\n  m.model.language_model exists: {type(mm.language_model).__name__}")

# Also check TinyLLaVA structure
print("\n\n=== TinyLLaVA ===")
try:
    from transformers import LlavaForConditionalGeneration as L2
    t = L2.from_pretrained(
        "bczhou/TinyLLaVA-3.1B", torch_dtype="auto", device_map="cpu",
        low_cpu_mem_usage=True, ignore_mismatched_sizes=True
    )
    print("Top-level named children:")
    for name, child in t.named_children():
        print(f"  {name}: {type(child).__name__}")

    if hasattr(t, 'model'):
        print("\nt.model named children:")
        for name, child in t.model.named_children():
            print(f"  {name}: {type(child).__name__}")
            if hasattr(child, '__len__'):
                print(f"    len={len(child)}")

        if hasattr(t.model, 'layers'):
            print(f"\n  t.model.layers: len={len(t.model.layers)}")
            if len(t.model.layers) > 0:
                layer0 = t.model.layers[0]
                print(f"  layer[0] type: {type(layer0).__name__}")
                for n, c in layer0.named_children():
                    print(f"    {n}: {type(c).__name__}")
except Exception as e:
    print(f"  Error: {e}")

print("\nDone.")
