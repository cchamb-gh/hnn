from model_api import load_model, sample_batch

inputs = sample_batch()

for kind in ["symmetric", "baseline"]:
    model = load_model(kind)
    model.eval()

    if isinstance(inputs, tuple):
        outputs = model(*inputs)
    else:
        outputs = model(inputs)

    print(kind, outputs.shape)