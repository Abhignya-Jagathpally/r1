"""Generate polished SVG figures for README."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ── Color palette (professional, accessible) ──
COLORS = {
    'primary': '#2563EB',      # blue-600
    'secondary': '#7C3AED',    # violet-600
    'success': '#059669',      # emerald-600
    'warning': '#D97706',      # amber-600
    'danger': '#DC2626',       # red-600
    'gray': '#6B7280',         # gray-500
    'light': '#F3F4F6',        # gray-100
    'dark': '#111827',         # gray-900
    'bg': '#FFFFFF',
}

STAGE_COLORS = ['#3B82F6', '#6366F1', '#8B5CF6', '#A855F7', 
                '#EC4899', '#F43F5E', '#F97316', '#EAB308', '#22C55E']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.edgecolor': 'white',
})


# ═══════════════════════════════════════════════════════════════════════
# Figure 1: Pipeline Architecture (horizontal fishbone)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(16, 5))
ax.set_xlim(-0.5, 9.5)
ax.set_ylim(-1.8, 2.5)
ax.axis('off')

stages = ['INGEST', 'CLEANSE', 'ENGINEER', 'SPLIT', 'BASELINES', 
          'ADVANCED', 'EVALUATE', 'AUTORESEARCH', 'REPORT']
icons = ['📥', '🧹', '⚙️', '✂️', '📊', '🧠', '📏', '🔬', '📋']
details = [
    'GDC API\n995 patients',
    'MICE impute\nWinsorize',
    'Slopes, SLiM-CRAB\nRolling windows',
    'Patient-level\nStratified',
    'LogReg, XGB\nCoxPH, RSF',
    'DeepHit, TFT\nMultimodal',
    'Bootstrap CI\nCalibration',
    'Optuna TPE\nLocked preproc',
    'Takeaways\nManifest'
]

# Draw spine
ax.annotate('', xy=(9.3, 0), xytext=(-0.3, 0),
            arrowprops=dict(arrowstyle='->', color='#D1D5DB', lw=3))

# Draw stages
for i, (stage, icon, detail, color) in enumerate(zip(stages, icons, details, STAGE_COLORS)):
    # Circle node
    circle = plt.Circle((i, 0), 0.38, color=color, ec='white', lw=2, zorder=3)
    ax.add_patch(circle)
    ax.text(i, 0, str(i), ha='center', va='center', fontsize=13, 
            fontweight='bold', color='white', zorder=4)
    
    # Stage name above
    ax.text(i, 0.7, stage, ha='center', va='bottom', fontsize=9, 
            fontweight='bold', color=COLORS['dark'])
    
    # Details below
    ax.text(i, -0.7, detail, ha='center', va='top', fontsize=7.5, 
            color=COLORS['gray'], linespacing=1.3)

# Title
ax.text(4.5, 2.2, 'MM Digital Twin — Fishbone Architecture', 
        ha='center', va='center', fontsize=16, fontweight='bold', color=COLORS['dark'])
ax.text(4.5, 1.8, '9 checkpointed stages · patient-level splits · frozen preprocessing', 
        ha='center', va='center', fontsize=10, color=COLORS['gray'])

# Checkpoint indicators
for i in range(9):
    ax.plot(i, -1.45, 's', color=STAGE_COLORS[i], markersize=6, zorder=3)
ax.text(4.5, -1.7, '■ = checkpoint (hash · shape · timing · params · metrics · git SHA)', 
        ha='center', va='center', fontsize=8, color=COLORS['gray'])

plt.tight_layout(pad=0.5)
plt.savefig('assets/pipeline_architecture.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('assets/pipeline_architecture.svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ pipeline_architecture.png/svg")


# ═══════════════════════════════════════════════════════════════════════
# Figure 2: Model Performance Comparison
# ═══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1.2, 1]})

# Left: AUROC bar chart with CI
models = ['LogReg', 'XGBoost', 'RSF']
val_auroc = [0.604, 0.857, 0.846]
test_auroc = [0.758, 0.703, 0.539]
test_ci_lo = [0.673, 0.621, 0.489]
test_ci_hi = [0.829, 0.780, 0.585]
test_err = [[np.array(test_auroc) - np.array(test_ci_lo)], 
            [np.array(test_ci_hi) - np.array(test_auroc)]]

x = np.arange(len(models))
w = 0.35

bars1 = ax1.bar(x - w/2, val_auroc, w, label='Validation', color='#93C5FD', ec='#3B82F6', lw=1)
bars2 = ax1.bar(x + w/2, test_auroc, w, label='Test', color='#FCD34D', ec='#F59E0B', lw=1,
                yerr=[np.array(test_auroc) - np.array(test_ci_lo), 
                      np.array(test_ci_hi) - np.array(test_auroc)],
                capsize=4, error_kw={'lw': 1.5, 'color': '#92400E'})

# Benchmark line
ax1.axhline(y=0.78, color='#DC2626', linestyle='--', lw=1.5, alpha=0.7, label='Benchmark (0.78)')
ax1.axhspan(0.76, 0.80, alpha=0.08, color='#DC2626')

ax1.set_ylabel('AUROC', fontsize=11, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=10)
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax1.set_title('Model Comparison (Tier 1 GDC Data)', fontsize=12, fontweight='bold', pad=12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='y', alpha=0.3)

# Right: Radar/metrics table
ax2.axis('off')
metrics_data = [
    ['', 'LogReg', 'XGBoost', 'RSF'],
    ['Test AUROC', '0.758', '0.703', '0.539'],
    ['95% CI', '[.67, .83]', '[.62, .78]', '[.49, .58]'],
    ['Brier', '0.140', '0.149', '0.268'],
    ['C-Index', '0.541', '0.529', '0.573'],
    ['ECE', '0.163', '0.099', '0.222'],
]

table = ax2.table(cellText=metrics_data[1:], colLabels=metrics_data[0],
                  cellLoc='center', loc='center',
                  colColours=['#F3F4F6', '#DBEAFE', '#FEF3C7', '#D1FAE5'])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.6)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#E5E7EB')
    if row == 0:
        cell.set_text_props(fontweight='bold')
        cell.set_facecolor('#F9FAFB')
    if col == 0 and row > 0:
        cell.set_text_props(fontweight='bold')
    # Highlight best AUROC
    if row == 1 and col == 1:
        cell.set_facecolor('#BBF7D0')

ax2.set_title('Evaluation Metrics (Test Set)', fontsize=12, fontweight='bold', pad=12)

plt.tight_layout(pad=2)
plt.savefig('assets/model_performance.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('assets/model_performance.svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ model_performance.png/svg")


# ═══════════════════════════════════════════════════════════════════════
# Figure 3: Data Flow (vertical)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

boxes = [
    (5, 9.2, 'RAW DATA', '995 patients · GDC Cases API', '#3B82F6'),
    (5, 7.8, 'INGEST', 'Column mapping · Type enforcement', '#6366F1'),
    (5, 6.4, 'CLEANSE', 'MICE imputation · Winsorization (train-only fit)', '#8B5CF6'),
    (5, 5.0, 'ENGINEER', 'Slopes · Rolling windows · SLiM-CRAB · Trajectory aggs', '#A855F7'),
    (5, 3.6, 'SPLIT', 'Patient-level stratified · No visit leakage', '#EC4899'),
]

for (cx, cy, title, sub, color) in boxes:
    rect = FancyBboxPatch((cx-2.8, cy-0.45), 5.6, 0.9, 
                           boxstyle="round,pad=0.1", 
                           facecolor=color, edgecolor='white', alpha=0.9, lw=2)
    ax.add_patch(rect)
    ax.text(cx, cy+0.08, title, ha='center', va='center', fontsize=12, 
            fontweight='bold', color='white')
    ax.text(cx, cy-0.22, sub, ha='center', va='center', fontsize=8.5, color='#E5E7EB')

# Arrows between boxes
for y_from, y_to in [(8.75, 8.25), (7.35, 6.85), (5.95, 5.45), (4.55, 4.05)]:
    ax.annotate('', xy=(5, y_to), xytext=(5, y_from),
                arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=2))

# Branch to baselines and advanced
ax.annotate('', xy=(2.8, 2.7), xytext=(4.2, 3.15),
            arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=2))
ax.annotate('', xy=(7.2, 2.7), xytext=(5.8, 3.15),
            arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=2))

# Baselines box
rect_b = FancyBboxPatch((0.5, 1.95), 4.2, 0.9,
                         boxstyle="round,pad=0.1",
                         facecolor='#F43F5E', edgecolor='white', alpha=0.9, lw=2)
ax.add_patch(rect_b)
ax.text(2.6, 2.48, 'BASELINES', ha='center', va='center', fontsize=12, 
        fontweight='bold', color='white')
ax.text(2.6, 2.18, 'LogReg · XGBoost · RSF · CoxPH', ha='center', va='center', 
        fontsize=8.5, color='#E5E7EB')

# Advanced box
rect_a = FancyBboxPatch((5.3, 1.95), 4.2, 0.9,
                         boxstyle="round,pad=0.1",
                         facecolor='#F97316', edgecolor='white', alpha=0.9, lw=2)
ax.add_patch(rect_a)
ax.text(7.4, 2.48, 'ADVANCED', ha='center', va='center', fontsize=12, 
        fontweight='bold', color='white')
ax.text(7.4, 2.18, 'DeepHit · TFT · Multimodal Fusion', ha='center', va='center', 
        fontsize=8.5, color='#E5E7EB')

# Converge to evaluate
ax.annotate('', xy=(4.2, 1.2), xytext=(2.6, 1.95),
            arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=2))
ax.annotate('', xy=(5.8, 1.2), xytext=(7.4, 1.95),
            arrowprops=dict(arrowstyle='->', color='#9CA3AF', lw=2))

# Evaluate box
rect_e = FancyBboxPatch((2.2, 0.3), 5.6, 0.9,
                         boxstyle="round,pad=0.1",
                         facecolor='#22C55E', edgecolor='white', alpha=0.9, lw=2)
ax.add_patch(rect_e)
ax.text(5, 0.83, 'EVALUATE & REPORT', ha='center', va='center', fontsize=12, 
        fontweight='bold', color='white')
ax.text(5, 0.53, 'Bootstrap AUROC · Brier · C-Index · Calibration · Autoresearch', 
        ha='center', va='center', fontsize=8.5, color='#E5E7EB')

plt.tight_layout(pad=0.5)
plt.savefig('assets/data_flow.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('assets/data_flow.svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ data_flow.png/svg")


# ═══════════════════════════════════════════════════════════════════════
# Figure 4: Stage Timing (horizontal bars)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(1, 1, figsize=(10, 4))

stages_timing = ['ingest', 'cleanse', 'engineer', 'split', 'baselines', 
                 'advanced', 'evaluate', 'autoresearch', 'report']
durations = [0.06, 0.07, 12.74, 0.05, 6.70, 11.61, 1.04, 5.26, 0.01]

y_pos = np.arange(len(stages_timing))[::-1]
bars = ax.barh(y_pos, durations, color=STAGE_COLORS, height=0.7, ec='white', lw=1)

for bar, dur in zip(bars, durations):
    if dur > 1:
        ax.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height()/2,
                f'{dur:.1f}s', ha='right', va='center', fontsize=9, 
                color='white', fontweight='bold')
    else:
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                f'{dur:.2f}s', ha='left', va='center', fontsize=9, color=COLORS['gray'])

ax.set_yticks(y_pos)
ax.set_yticklabels(stages_timing, fontsize=10)
ax.set_xlabel('Duration (seconds)', fontsize=11)
ax.set_title('Pipeline Stage Execution Profile', fontsize=13, fontweight='bold', pad=12)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3)

total = sum(durations)
ax.text(0.98, 0.02, f'Total: {total:.1f}s', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=10, color=COLORS['gray'],
        fontweight='bold')

plt.tight_layout(pad=1)
plt.savefig('assets/stage_timing.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.savefig('assets/stage_timing.svg', bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print("✓ stage_timing.png/svg")

print("\nAll figures generated!")