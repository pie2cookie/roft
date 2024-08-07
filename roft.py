class RoFTLinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 4,
        init_gradient_allign: bool = False,
    ):
        super(RoFTLinear, self).__init__()
        self.base_layer = copy.deepcopy(base_layer)
        self.r = r
        self.roft_B = nn.ParameterList([nn.Parameter(torch.empty(base_layer.in_features // 2, dtype=base_layer.weight.dtype)) for _ in range(r)])
        for rB in self.roft_B:
            # nn.init.zeros_(rB)
            nn.init.normal_(rB, mean=0.0, std=0.02)
        # if init_gradient_allign:
             # nn.init.normal_(self.roft_B, mean=0.0, std=0.02)
        # else:
             # for rB in self.roft_B:
             #     nn.init.normal_(rB, mean=0.0, std=0.02)
        for param in self.base_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur_cnt = self.base_layer.in_features
        step_size = 1
        rot_list = []
        idxs = []
        l_theta = self.base_layer.in_features // 2
        idxs.append(torch.arange(l_theta) * 2)
        mask = [torch.zeros(l_theta, dtype=torch.bool) for _ in range(4)]
        for i in range(l_theta):
            if i % 2 ==1:   mask[1][i] = True
            if i % 4 > 1:   mask[2][i] = True
            if i % 8 > 3:   mask[3][i] = True
        idxs.append(idxs[0] -1 * mask[1])
        idxs.append(idxs[1] -2 * mask[2])
        idxs.append(idxs[2] -4 * mask[3])
        rot_weight = torch.eye(self.base_layer.in_features).to(x.device)
        for i in range(self.r):
            if cur_cnt<2:
                break
            cur_cnt = cur_cnt // 2
            rot_mat = torch.eye(self.base_layer.in_features).to(x.device)
            all_theta = self.roft_B[i].float() * torch.pi
            idx =idxs[i]
            rot_mat[idx, idx] = torch.cos(all_theta)
            rot_mat[idx, idx+step_size] = -torch.sin(all_theta)
            rot_mat[idx+step_size, idx] = torch.sin(all_theta)
            rot_mat[idx+step_size, idx+step_size] = torch.cos(all_theta)
            rot_weight = torch.mm(rot_mat.to_sparse(), rot_weight)
            step_size=step_size * 2
        
        result = F.linear(x, rot_weight.to(self.base_layer.weight.dtype))
        result = self.base_layer(result)

        return result
