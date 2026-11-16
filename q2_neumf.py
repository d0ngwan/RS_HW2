import sys
import numpy as np, pandas as pd, torch, torch.nn as nn

SETTINGS = {'a': (40, 128, (256, 128, 64), 4, 0.5, 25), 'b': (20, 128, (256, 128, 64), 4, 0.5, 20),
            'c': (40, 64, (128, 64, 32), 4, 0.5, 40), 'd': (40, 128, (256, 128, 64), 8, 0.5, 13),
            'e': (30, 128, (256, 128, 64), 4, 0.3, 25), 'f': (40, 128, (256, 128, 64), 16, 0.5, 8),
            'g': (60, 128, (256, 128, 64), 4, 0.5, 30)}
SEEDS, LR, BS = 10, 1e-3, 1024
dev = 'cuda'
tr = pd.read_csv('data/train.csv').sort_values(['userId', 'time_idx']).reset_index(drop=True)
te = pd.read_csv('data/test.csv')
NU, NI = 610, int(tr.movieId.max()) + 1
DISC = 1 / torch.log2(torch.arange(2, 22, device=dev).float()); IDCG = torch.cumsum(DISC, 0)

class View:
    def __init__(s, df):
        s.u = torch.as_tensor(df.userId.values, device=dev); s.i = torch.as_tensor(df.movieId.values, device=dev)
        s.r = torch.as_tensor(df.rating.values, dtype=torch.float32, device=dev)
        s.n = torch.as_tensor(df.groupby('userId').movieId.transform('size').values, dtype=torch.float32, device=dev)
        s.pos = torch.as_tensor(df.groupby('userId').cumcount().values, dtype=torch.float32, device=dev)
        s.rev = s.n - 1 - s.pos                                           # 0 = most recent rating
        s.seen = torch.zeros(NU, NI, dtype=torch.bool, device=dev); s.seen[s.u, s.i] = True
        s.ulen = s.seen.sum(1).float()

class NeuMF(nn.Module):
    def __init__(s, nuser, d, layers):
        super().__init__()
        s.ug, s.ig = nn.Embedding(nuser, d), nn.Embedding(NI, d); s.um, s.im = nn.Embedding(nuser, layers[0] // 2), nn.Embedding(NI, layers[0] // 2)
        mlp = []
        for a, b in zip(layers[:-1], layers[1:]): mlp += [nn.Linear(a, b), nn.ReLU()]
        s.mlp = nn.Sequential(*mlp); s.out = nn.Linear(d + layers[-1], 1)
        for e in [s.ug, s.ig, s.um, s.im]: nn.init.normal_(e.weight, std=0.01)
    def forward(s, u, i): return s.out(torch.cat([s.ug(u) * s.ig(i), s.mlp(torch.cat([s.um(u), s.im(i)], -1))], -1)).squeeze(-1)
    @torch.no_grad()
    def scores_for(s, uid):
        items = torch.arange(NI, device=dev); out = []
        for b in range(0, len(uid), 32):
            u = uid[b:b + 32]; out.append(s(u[:, None].expand(-1, NI).reshape(-1), items[None, :].expand(len(u), -1).reshape(-1)).reshape(len(u), NI))
        return torch.cat(out)

def windows(V, L):
    S = max(L // 2, 1); rev = V.rev.long()
    kmax = torch.div((V.ulen.long().clamp(min=1) - 1), S, rounding_mode='floor') + 1
    base = NU + torch.cumsum(torch.cat([torch.zeros(1, device=dev, dtype=torch.long), kmax[:-1]]), 0)
    k_hi = torch.div(rev, S, rounding_mode='floor'); k_lo = torch.div((rev - L).clamp(min=-1) + S, S, rounding_mode='floor').clamp(min=0)
    us, its, wid = [], [], []
    for d in range(0, (L + S - 1) // S):
        k = k_hi - d; ok = (k >= k_lo) & (k >= 0) & (k < kmax[V.u])
        us.append(V.u[ok]); its.append(V.i[ok]); wid.append(base[V.u[ok]] + k[ok])
    return torch.cat(us), torch.cat(its), torch.cat(wid), base, int(NU + kmax.sum().item())

def train(V, seed, L, dim, layers, neg, tau, epochs):
    torch.manual_seed(seed); like = V.r >= 4
    wu, wi, wid, base, nuser = windows(V, L)
    lk = torch.zeros(NU, NI, dtype=torch.bool, device=dev); lk[V.u[like], V.i[like]] = True
    wm = lk[wu, wi]; wu, wi, wid = wu[wm], wi[wm], wid[wm]
    pu_model = torch.cat([V.u[like], wid]); pu_real = torch.cat([V.u[like], wu]); pi = torch.cat([V.i[like], wi])
    wr = torch.exp(-V.rev / (tau * V.n))[like]; wr = wr / wr.mean()
    w = torch.cat([wr, torch.ones(len(wid), device=dev)]); n = len(pi)
    m = NeuMF(nuser, dim, list(layers)).to(dev); opt = torch.optim.Adam(m.parameters(), lr=LR); bce = nn.functional.binary_cross_entropy_with_logits
    for ep in range(epochs):
        m.train(); nm_ = pu_model.repeat(neg); nr_ = pu_real.repeat(neg); ni_ = torch.randint(0, NI, (n * neg,), device=dev); keep = ~V.seen[nr_, ni_]; k = int(keep.sum())
        uu = torch.cat([pu_model, nm_[keep]]); ii = torch.cat([pi, ni_[keep]]); yy = torch.cat([torch.ones(n, device=dev), torch.zeros(k, device=dev)]); ww = torch.cat([w, torch.ones(k, device=dev)])
        perm = torch.randperm(len(yy), device=dev)
        for b in range(0, len(yy), BS):
            idx = perm[b:b + BS]; loss = (bce(m(uu[idx], ii[idx]), yy[idx], reduction='none') * ww[idx]).mean(); opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return m.scores_for(torch.arange(NU, device=dev)), m.scores_for(base)

def zs(S): return (S - S.mean(1, keepdim=True)) / S.std(1, keepdim=True).clamp(min=1e-9)

def ensemble(V, seeds):
    total = torch.zeros(NU, NI, device=dev)
    for name, (L, dim, layers, neg, tau, epochs) in SETTINGS.items():
        R = torch.zeros(NU, NI, device=dev); W = torch.zeros(NU, NI, device=dev)
        for s in range(seeds):
            r, w = train(V, s, L, dim, layers, neg, tau, epochs); R += r; W += w
        total += zs(zs(R) + zs(W)); print(f'  setting {name} done', flush=True)
    return total.masked_fill(V.seen, -1e30)

def cut(df, frac=0.2):
    n = df.groupby('userId').movieId.transform('size').values; pos = df.groupby('userId').cumcount().values
    k = np.minimum(np.maximum(1, np.round(n * frac)), n - 1).astype(int); hold = pos >= n - k
    return df[~hold], df[hold]

with torch.enable_grad():
    if '--validate' in sys.argv:
        cur = tr
        for fold in 'AB':
            base, hold = cut(cur); V = View(base); S = ensemble(V, 5)
            liked = hold[hold.rating >= 4]; Y = torch.zeros(NU, NI, device=dev); Y[liked.userId.values, liked.movieId.values] = 1
            users = torch.zeros(NU, dtype=torch.bool, device=dev); users[torch.as_tensor(te.userId.values, device=dev)] = True; users &= Y.sum(1) > 0
            hit = torch.gather(Y, 1, torch.topk(S, 20, dim=1).indices); nd = (hit * DISC).sum(1) / IDCG[(Y.sum(1).clamp(1, 20) - 1).long()]
            print(f'fold {fold}: NDCG@20 (liked) {nd[users].mean().item():.4f}', flush=True); cur = base
    S = ensemble(View(tr), SEEDS)
top = torch.topk(S, 20, dim=1).indices.cpu().numpy()
pd.DataFrame({'id': te.id.values, 'recommendations': [' '.join(map(str, top[u])) for u in te.userId.values]}).to_csv('submission_q2_neumf.csv', index=False)