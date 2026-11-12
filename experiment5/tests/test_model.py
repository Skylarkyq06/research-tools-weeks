import torch

from experiment5.src.model import MLP, build_initial_model, get_parameters, set_parameters


def main():
    m = build_initial_model(42)
    x = torch.randn(4, 1, 28, 28)
    assert m(x).shape == (4, 10)
    params = get_parameters(m)
    m2 = MLP()
    set_parameters(m2, params)
    assert torch.allclose(m(x), m2(x))
    print("forward shape OK, params:", sum(p.numel() for p in m.parameters()))


if __name__ == "__main__":
    main()
