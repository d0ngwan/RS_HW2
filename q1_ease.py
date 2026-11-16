import sys
import numpy as np, pandas as pd, torch

WINDOW, FULL, LAMBDA, TAU = 80, 0.3, 200.0, 0.35
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_grad_enabled(False)
tr = pd.read_csv('data/train.csv').sort_values(['userId', 'time_idx']).reset_index(drop=True)
te = pd.read_csv('data/test.csv')
NU, NI = int(tr.userId.max()) + 1, int(tr.movieId.max()) + 1

def cut(df, frac=0.2):
    n = df.groupby('userId').movieId.transform('size').values; pos = df.groupby('userId').cumcount().values
    k = np.minimum(np.maximum(1, np.round(n * frac)), n - 1).astype(int); hold = pos >= n - k
    return df[~hold], df[hold]

def ease_scores(df):
    u = torch.as_tensor(df.userId.values, device=dev); i = torch.as_tensor(df.movieId.values, device=dev)
    r = torch.as_tensor(df.rating.values, dtype=torch.float32, device=dev)
    n = torch.as_tensor(df.groupby('userId').movieId.transform('size').values, dtype=torch.float32, device=dev)
    pos = torch.as_tensor(df.groupby('userId').cumcount().values, dtype=torch.float32, device=dev); age = n - 1 - pos
    X = torch.zeros(NU, NI, device=dev); X[u, i] = 1.0; pop = X.sum(0)
    # Gram of the window rows: pairs (earlier item, later item) of the same user, gap d < WINDOW, weight (WINDOW - d) / WINDOW
    W = torch.zeros(NI, NI, device=dev)
    for d in range(1, WINDOW):
        same = u[:-d] == u[d:]
        W.index_put_((i[:-d][same], i[d:][same]), torch.full((int(same.sum()),), (WINDOW - d) / WINDOW, device=dev), accumulate=True)
    G = FULL * (X.T @ X) + (1 - FULL) * (W + W.T + torch.diag(pop)); del W
    G.diagonal().add_(LAMBDA); P = torch.linalg.inv(G); del G
    B = P / (-P.diagonal())[None, :]; B.fill_diagonal_(0)
    h = torch.zeros(NU, NI, device=dev); h[u, i] = (r >= 4).float() * torch.exp(-age / (TAU * n))      # liked items, recency-weighted
    return h @ B, X > 0

def top20(S, seen): return torch.topk(S.masked_fill(seen, -1e30), 20, dim=1).indices

def ndcg20(top, hold):
    disc = 1 / torch.log2(torch.arange(2, 22, device=dev).float()); idcg = torch.cumsum(disc, 0)
    Y = torch.zeros(NU, NI, device=dev); lk = hold[hold.rating >= 4]; Y[lk.userId.values, lk.movieId.values] = 1
    cnt = Y.sum(1); users = torch.zeros(NU, dtype=torch.bool, device=dev); users[te.userId.values] = True; users &= cnt > 0
    dcg = (torch.gather(Y, 1, top) * disc[None]).sum(1)
    return (dcg / idcg[(cnt.clamp(1, 20) - 1).long()])[users].mean().item()

if '--validate' in sys.argv:
    cur = tr; vals = []
    for name in 'ABC':
        base, hold = cut(cur); S, seen = ease_scores(base); vals.append(ndcg20(top20(S, seen), hold)); cur = base
        print(f'fold {name}: NDCG@20 (liked) = {vals[-1]:.4f}')
    print(f'mean of the three folds: {np.mean(vals):.4f}')

S, seen = ease_scores(tr); top = top20(S, seen).cpu().numpy()
sub = pd.DataFrame({'id': te.id.values, 'recommendations': [' '.join(map(str, top[u])) for u in te.userId.values]})
sub.to_csv('submission_q1_ease.csv', index=False)