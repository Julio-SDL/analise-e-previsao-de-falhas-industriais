import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.ensemble import RandomForestClassifier


# configuracao visual

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.facecolor'   : '#1a1a2e',
    'figure.facecolor' : '#0f0f1a',
    'axes.edgecolor'   : '#444466',
    'axes.labelcolor'  : '#ccccdd',
    'axes.titlecolor'  : '#ffffff',
    'xtick.color'      : '#aaaacc',
    'ytick.color'      : '#aaaacc',
    'grid.color'       : '#2a2a4a',
    'grid.linestyle'   : '--',
    'grid.alpha'       : 0.5,
    'text.color'       : '#ccccdd',
})

COR_NORMAL = '#4fc3f7'
COR_FALHA  = '#ff5252'
COR_AVISO  = '#ffca28'


# carregamento e tratamento dos dados

print("Carregando dados e treinando modelo...")
df = pd.read_csv("dataset_sensores_industriais_200_registros.csv")
df = df.fillna(df.median(numeric_only=True))
df['termica_pressao']    = df['temperatura'] * df['pressao']
df['desgaste_acumulado'] = df['vibracao']    * df['tempo_operacao']


# treinamento do modelo

X = df.drop('falha', axis=1)
y = df['falha']

modelo = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
modelo.fit(X, y)

importancias = pd.Series(modelo.feature_importances_, index=X.columns).sort_values()
falhas_idx   = df[df['falha'] == 1].index
normais_idx  = df[df['falha'] == 0].index
n_falhas     = len(falhas_idx)

print("Gerando dashboard...")


# layout da figura — grade 3x2

fig = plt.figure(figsize=(20, 14))
fig.suptitle(
    'DASHBOARD DE MONITORAMENTO INDUSTRIAL — ANALISE DE SENSORES E FALHAS',
    fontsize=15, fontweight='bold', color='white', y=0.98
)

gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax4 = fig.add_subplot(gs[2, 0])
ax5 = fig.add_subplot(gs[2, 1])


# grafico 1 — historico de temperatura com falhas marcadas

ax1.plot(df.index, df['temperatura'], color=COR_NORMAL, linewidth=1.2,
         label='Temperatura (C)', zorder=2)

ax1.scatter(falhas_idx, df.loc[falhas_idx, 'temperatura'],
            color=COR_FALHA, s=45, zorder=3, label='Registro de Falha')

limiar = df['temperatura'].quantile(0.75)
ax1.axhline(limiar, color=COR_AVISO, linewidth=1.2, linestyle='--',
            label=f'Limiar de atencao (P75 = {limiar:.1f}C)')

ax1.fill_between(df.index, limiar, df['temperatura'],
                 where=df['temperatura'] > limiar,
                 alpha=0.15, color=COR_AVISO)

ax1.annotate(f'{n_falhas} falhas registradas',
             xy=(falhas_idx[0], df.loc[falhas_idx[0], 'temperatura']),
             xytext=(15, 20), textcoords='offset points',
             color=COR_FALHA, fontsize=8,
             arrowprops=dict(arrowstyle='->', color=COR_FALHA, lw=1.2))

ax1.set_title('Historico de Temperatura — Falhas Destacadas', fontsize=12, fontweight='bold')
ax1.set_xlabel('Registros (indice temporal)')
ax1.set_ylabel('Temperatura (C)')
ax1.legend(loc='upper right', fontsize=8, framealpha=0.3)
ax1.grid(True)


# grafico 2 — importancia das variaveis

cores_barras = [COR_FALHA if v >= importancias.quantile(0.75)
                else COR_AVISO if v >= importancias.quantile(0.40)
                else COR_NORMAL
                for v in importancias.values]

bars = ax2.barh(importancias.index, importancias.values,
                color=cores_barras, edgecolor='none', height=0.65)

for bar, val in zip(bars, importancias.values):
    ax2.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
             f'{val*100:.1f}%', va='center', fontsize=8, color='white')

legenda_imp = [
    Patch(color=COR_FALHA,  label='Alto impacto'),
    Patch(color=COR_AVISO,  label='Medio impacto'),
    Patch(color=COR_NORMAL, label='Baixo impacto'),
]

ax2.set_title('Importancia dos Sensores na Previsao de Falhas', fontsize=11, fontweight='bold')
ax2.set_xlabel('Grau de Impacto')
ax2.set_xlim(0, importancias.max() * 1.25)
ax2.legend(handles=legenda_imp, fontsize=7, framealpha=0.3, loc='lower right')
ax2.grid(axis='x')


# grafico 3 — scatter temperatura x pressao

ax3.scatter(df.loc[normais_idx, 'temperatura'], df.loc[normais_idx, 'pressao'],
            c=COR_NORMAL, alpha=0.5, s=25, label='Normal', zorder=2)
ax3.scatter(df.loc[falhas_idx, 'temperatura'], df.loc[falhas_idx, 'pressao'],
            c=COR_FALHA, alpha=0.85, s=50, label='Falha', zorder=3, marker='X')

lim_t = df['temperatura'].quantile(0.75)
lim_p = df['pressao'].quantile(0.75)
ax3.axvline(lim_t, color=COR_AVISO, linewidth=1, linestyle='--', alpha=0.7)
ax3.axhline(lim_p, color=COR_AVISO, linewidth=1, linestyle='--', alpha=0.7)

ax3.fill_betweenx([lim_p, df['pressao'].max() * 1.02],
                  lim_t, df['temperatura'].max() * 1.02,
                  alpha=0.08, color=COR_FALHA)
ax3.text(lim_t + 0.5, lim_p + 0.3, 'Zona de\nRisco', color=COR_FALHA, fontsize=7, alpha=0.9)

ax3.set_title('Temperatura x Pressao por Status', fontsize=11, fontweight='bold')
ax3.set_xlabel('Temperatura (C)')
ax3.set_ylabel('Pressao (bar)')
ax3.legend(fontsize=8, framealpha=0.3)
ax3.grid(True)


# grafico 4 — boxplot vibracao por status

dados_box  = [df.loc[normais_idx, 'vibracao'], df.loc[falhas_idx, 'vibracao']]
labels_box = ['Normal', 'Falha']
cores_box  = [COR_NORMAL, COR_FALHA]

bp = ax4.boxplot(dados_box, labels=labels_box, patch_artist=True,
                 medianprops=dict(color='white', linewidth=2),
                 whiskerprops=dict(color='#aaaacc'),
                 capprops=dict(color='#aaaacc'),
                 flierprops=dict(marker='o', markersize=4, alpha=0.5))

for patch, cor in zip(bp['boxes'], cores_box):
    patch.set_facecolor(cor)
    patch.set_alpha(0.5)

for i, dados in enumerate(dados_box, 1):
    mediana = np.median(dados)
    ax4.text(i, mediana + 0.01, f'  med={mediana:.2f}', color='white', fontsize=8, va='bottom')

ax4.set_title('Distribuicao de Vibracao — Normal vs Falha', fontsize=11, fontweight='bold')
ax4.set_ylabel('Vibracao (mm/s)')
ax4.grid(axis='y')


# grafico 5 — histograma desgaste acumulado

bins = np.linspace(df['desgaste_acumulado'].min(), df['desgaste_acumulado'].max(), 25)

ax5.hist(df.loc[normais_idx, 'desgaste_acumulado'], bins=bins,
         color=COR_NORMAL, alpha=0.6, label='Normal', edgecolor='none')
ax5.hist(df.loc[falhas_idx, 'desgaste_acumulado'], bins=bins,
         color=COR_FALHA, alpha=0.7, label='Falha', edgecolor='none')

limiar_deg = df['desgaste_acumulado'].quantile(0.90)
ax5.axvline(limiar_deg, color=COR_AVISO, linewidth=1.5, linestyle='--',
            label=f'P90 = {limiar_deg:.0f}')

ax5.set_title('Desgaste Acumulado (Vibracao x Tempo)', fontsize=11, fontweight='bold')
ax5.set_xlabel('Indice de Desgaste Acumulado')
ax5.set_ylabel('Frequencia')
ax5.legend(fontsize=8, framealpha=0.3)
ax5.grid(axis='y')


# rodape com estatisticas rapidas

tx_falha = n_falhas / len(df) * 100
med_temp = df['temperatura'].median()
med_vib  = df['vibracao'].median()

rodape = (f"Total de registros: {len(df)}   |   Falhas: {n_falhas} ({tx_falha:.1f}%)   |   "
          f"Mediana temperatura: {med_temp:.1f}C   |   Mediana vibracao: {med_vib:.2f} mm/s   |   "
          f"Variavel mais critica: {importancias.idxmax()}")

fig.text(0.5, 0.01, rodape, ha='center', fontsize=8, color='#888899', style='italic')


# salvar e exibir

plt.savefig('dashboard_industrial.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Dashboard salvo como 'dashboard_industrial.png'")
plt.show()
print("Concluido.")