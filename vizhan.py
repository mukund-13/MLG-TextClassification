import torch
from torchviz import make_dot
from han_model import HANModel

# Define dummy variables for model construction
in_channels_dict = {
    'document': 64,
    'author': 64,
    'tag': 64
}
out_channels = 4
metadata = (['document', 'author', 'tag'],
            [('document', 'to', 'author'),
             ('author', 'to', 'document'),
             ('document', 'to', 'tag'),
             ('tag', 'to', 'document')])

# Instantiate the model
model = HANModel(in_channels_dict, out_channels, metadata, hidden_channels=64, heads=1)
model.eval()

# Create dummy input tensors
x_dict = {
    'document': torch.randn(31632, 64),
    'author': torch.randn(2000, 64),
    'tag': torch.randn(4000, 64)
}
edge_index_dict = {
    ('document', 'to', 'author'): torch.randint(0,31632,(2,10000)),
    ('author', 'to', 'document'): torch.randint(0,2000,(2,10000)),
    ('document', 'to', 'tag'): torch.randint(0,31632,(2,10000)),
    ('tag', 'to', 'document'): torch.randint(0,4000,(2,10000))
}

# Run a forward pass
out = model(x_dict, edge_index_dict)

# Visualize the model
dot = make_dot(out, params=dict(model.named_parameters()))
dot.render("han_model_architecture", format="png")
