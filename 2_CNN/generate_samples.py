"""
Generate synthetic fruit sample images WITHOUT sklearn dependency.
Saves: synthetic_fruit_samples.png, apple_pixel.png, banana_pixel.png, orange_pixel.png
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import os

np.random.seed(42)

# ── Same make_images() logic as the notebook ─────────────
def make_images(label, n=40):
    imgs = []
    for _ in range(n):
        img = np.random.uniform(0, 0.1, (8, 8, 3))
        if label == 0:    # Apple
            img[:,:,0] = np.random.uniform(0.8, 1.0, (8,8))
        elif label == 1:  # Banana
            img[:,:,0] = np.random.uniform(0.8, 1.0, (8,8))
            img[:,:,1] = np.random.uniform(0.8, 1.0, (8,8))
        else:             # Orange
            img[:,:,0] = np.random.uniform(0.8, 1.0, (8,8))
            img[:,:,1] = np.random.uniform(0.4, 0.6, (8,8))
        imgs.append(img)
    return np.array(imgs)

X = np.vstack([make_images(0), make_images(1), make_images(2)])
y = np.array([0]*40 + [1]*40 + [2]*40)

# Manual split (no sklearn)
np.random.seed(42)
indices = np.random.permutation(len(X))
split   = int(0.8 * len(X))
train_idx, test_idx = indices[:split], indices[split:]
X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

print(f"Images shape : {X.shape}")
print(f"Train        : {len(X_train)}  |  Test : {len(X_test)}")

OUT = os.path.dirname(os.path.abspath(__file__))

# ── Big comparison figure ─────────────────────────────────
labels  = ['Apple', 'Banana', 'Orange']
emojis  = ['(A)', '(B)', '(O)']
colors  = ['#e74c3c', '#d4ac0d', '#e67e22']

fig, axes = plt.subplots(2, 3, figsize=(11, 7))
fig.patch.set_facecolor('#1a1a2e')

fig.suptitle(
    'What the CNN Actually Sees  —  Synthetic 8×8 Pixel Images',
    fontsize=14, fontweight='bold', color='white', y=1.01
)

for col, (fruit_label, title, emoji, color) in enumerate(
        zip([0, 1, 2], labels, emojis, colors)):

    idx    = np.where(y == fruit_label)[0][0]
    sample = np.clip(X[idx], 0, 1)

    # ── Top row: 8×8 pixel image ────────────────────────
    ax_top = axes[0, col]
    ax_top.imshow(sample, interpolation='nearest', aspect='equal')
    ax_top.set_title(f'{emoji} {title}', fontsize=14, fontweight='bold',
                     color=color, pad=10)
    ax_top.set_xlabel('8 × 8 RGB pixels', fontsize=9, color='#aaaaaa')
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    ax_top.set_facecolor('#0d0d1a')
    for spine in ax_top.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

    # ── Bottom row: channel bar chart ──────────────────
    ax_bot = axes[1, col]
    ax_bot.set_facecolor('#12122a')
    means       = [sample[:,:,c].mean() for c in range(3)]
    bar_colors  = ['#e74c3c', '#2ecc71', '#3498db']
    ch_labels   = ['Red', 'Green', 'Blue']
    bars = ax_bot.bar(ch_labels, means, color=bar_colors,
                      width=0.5, edgecolor='#ffffff22', linewidth=1)
    ax_bot.set_ylim(0, 1.2)
    ax_bot.set_ylabel('Mean intensity', fontsize=9, color='#cccccc')
    ax_bot.set_title('Channel values', fontsize=10, color='#aaaaaa')
    ax_bot.tick_params(colors='#aaaaaa')
    for spine in ax_bot.spines.values():
        spine.set_edgecolor('#333355')
    for bar, val in zip(bars, means):
        ax_bot.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.04,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold', color='white')

plt.tight_layout()
out_path = os.path.join(OUT, 'synthetic_fruit_samples.png')
fig.savefig(out_path, dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none')
plt.close(fig)
print(f'Saved -> {out_path}')

# ── Individual pixel previews (for README table) ─────────
for fruit_label, name in zip([0, 1, 2],
                              ['apple_pixel', 'banana_pixel', 'orange_pixel']):
    idx    = np.where(y == fruit_label)[0][0]
    sample = np.clip(X[idx], 0, 1)

    fig2, ax2 = plt.subplots(1, 1, figsize=(2, 2))
    fig2.patch.set_facecolor('#1a1a2e')
    ax2.imshow(sample, interpolation='nearest')
    ax2.axis('off')
    p = os.path.join(OUT, f'{name}.png')
    fig2.savefig(p, dpi=100, bbox_inches='tight',
                 facecolor='#1a1a2e', pad_inches=0.05)
    plt.close(fig2)
    print(f'Saved -> {p}')

print('\nAll done!')
