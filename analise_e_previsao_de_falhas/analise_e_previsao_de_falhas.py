import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from google import genai
from dotenv import load_dotenv


# configuracao da api

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
gemini_client = None
GEMINI_MODEL  = "gemini-2.5-flash-lite"   # modelo atual (mai/2026)

if GOOGLE_API_KEY:
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)
else:
    print("⚠️  AVISO: GOOGLE_API_KEY não encontrada. As opções 2 e 3 do menu não estarão disponíveis.")


# carregamento e tratamento de dados

print("\nCarregando e tratando os dados...")
df = pd.read_csv("dataset_sensores_industriais_200_registros.csv")

df = df.fillna(df.median(numeric_only=True))


# feature engineering

df['termica_pressao']    = df['temperatura'] * df['pressao']
df['desgaste_acumulado'] = df['vibracao']    * df['tempo_operacao']


# treinamento do modelo

print("Treinando o modelo Random Forest...")

FEATURES = [c for c in df.columns if c != 'falha']
TARGET   = 'falha'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelo_rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
modelo_rf.fit(X_train, y_train)

y_pred   = modelo_rf.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)
print(f"Modelo treinado com sucesso! Acurácia: {acuracia:.2%}")

# percentis por feature (base para thresholds de severidade)
PERCENTIS = {col: df[col].quantile([0.25, 0.50, 0.75]).to_dict()
             for col in FEATURES}

# importâncias do modelo
IMPORTANCIAS = dict(zip(FEATURES, modelo_rf.feature_importances_))

# correlação de cada feature com falha
CORRELACOES = df[FEATURES + [TARGET]].corr()[TARGET].drop(TARGET).abs()

# relatorio tecnico (sem ia - gerado na inicialização)

def gerar_relatorio_tecnico():
    linha = "=" * 62
    print(f"\n{linha}")
    print("  📋  RELATÓRIO TÉCNICO DE MANUTENÇÃO — VISÃO DO TÉCNICO")
    print(f"  Data/hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(linha)

    total     = len(df)
    n_falhas  = int(df[TARGET].sum())
    tx_falha  = n_falhas / total * 100

    print(f"\n  🏭  VISÃO GERAL")
    print(f"     Registros analisados : {total}")
    print(f"     Falhas registradas   : {n_falhas}  ({tx_falha:.1f}%)")

    print(f"\n  🔴  O QUE MAIS DESGASTA A MÁQUINA (por importância no modelo)")
    ranking = sorted(IMPORTANCIAS.items(), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(ranking, 1):
        corr = CORRELACOES.get(feat, 0)
        barra = "█" * int(imp * 40)
        print(f"     {i:2}. {feat:<22} imp={imp:.3f}  corr_falha={corr:.3f}  {barra} {imp*100:.1f}%")

    print(f"\n  🟢  O QUE MENOS IMPACTA NO DESGASTE")
    menos = [(feat, imp) for feat, imp in ranking if imp < 0.05]
    if menos:
        for feat, imp in menos:
            print(f"       * {feat:<22} imp={imp:.3f}")
    else:
        ultimo_feat, ultimo_imp = ranking[-1]
        print(f"       Todas as variaveis tem impacto relevante no modelo (todas acima de 5%).")
        print(f"       A de menor influencia e '{ultimo_feat}' com {ultimo_imp*100:.1f}% de importancia.")

    print(f"\n  ⚠️   SITUAÇÕES PERIGOSAS IDENTIFICADAS")


    # alta pressão + alta temperatura
    lim_temp  = PERCENTIS['temperatura'][0.75]
    lim_press = PERCENTIS['pressao'][0.75]
    risco_tp  = df[(df['temperatura'] > lim_temp) & (df['pressao'] > lim_press)]
    tx_risco_tp = risco_tp[TARGET].mean() * 100 if len(risco_tp) else 0
    print(f"     🔥 Temperatura ALTA (>{lim_temp:.1f}) + Pressão ALTA (>{lim_press:.1f})")
    print(f"        Registros nessa condição: {len(risco_tp)}  |  Taxa de falha: {tx_risco_tp:.1f}%")


    # alta vibração + longa operação
    lim_vib  = PERCENTIS['vibracao'][0.75]
    lim_top  = PERCENTIS['tempo_operacao'][0.75]
    risco_vo = df[(df['vibracao'] > lim_vib) & (df['tempo_operacao'] > lim_top)]
    tx_risco_vo = risco_vo[TARGET].mean() * 100 if len(risco_vo) else 0
    print(f"     💥 Vibração ALTA (>{lim_vib:.2f}) + Tempo operação longo (>{lim_top:.0f}h)")
    print(f"        Registros nessa condição: {len(risco_vo)}  |  Taxa de falha: {tx_risco_vo:.1f}%")


    # desgaste acumulado extremo
    lim_deg = df['desgaste_acumulado'].quantile(0.90)
    risco_d = df[df['desgaste_acumulado'] > lim_deg]
    tx_risco_d = risco_d[TARGET].mean() * 100 if len(risco_d) else 0
    print(f"     ⚙️  Desgaste acumulado extremo (>{lim_deg:.1f})")
    print(f"        Registros nessa condição: {len(risco_d)}  |  Taxa de falha: {tx_risco_d:.1f}%")


    # --- estatisticas dos sensores --- 
    print(f"\n  📊  ESTATÍSTICAS DOS SENSORES (base histórica)")
    stats = df[FEATURES].describe().T[['50%','std','min','25%','75%','max']]
    print(f"     {'Sensor':<22} {'Mediana':>8} {'Std':>8} {'Min':>8} {'P25':>8} {'P75':>8} {'Max':>8}")
    print(f"     {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for feat, row in stats.iterrows():
        print(f"     {feat:<22} {row['50%']:>8.2f} {row['std']:>8.2f} "
              f"{row['min']:>8.2f} {row['25%']:>8.2f} {row['75%']:>8.2f} {row['max']:>8.2f}")


    # --- recomendacoes --- 
    top3 = [f for f, _ in ranking[:3]]
    print(f"\n  🛠️   RECOMENDAÇÕES DO TÉCNICO")
    print(f"     1. Monitorar prioritariamente: {', '.join(top3)}")
    print(f"     2. Definir alarmes quando temperatura + pressão excederem "
          f"{lim_temp:.0f}°C e {lim_press:.0f} bar simultaneamente.")
    print(f"     3. Agendar manutenção preventiva quando tempo_operacao > {lim_top:.0f}h.")
    print(f"     4. Inspecionar rolamentos/vedações ao detectar vibração > {lim_vib:.2f} mm/s.")
    print(f"\n{linha}\n")


# funcoes auxiliares — severidade por variável

def severidade(valor, feat):
    """Retorna emoji + texto de severidade baseado nos percentis históricos."""
    p25 = PERCENTIS[feat][0.25]
    p75 = PERCENTIS[feat][0.75]

    # features onde valor ALTO = ruim (desgaste)
    alto_ruim = {
        'temperatura', 'pressao', 'vibracao', 'tempo_operacao',
        'termica_pressao', 'desgaste_acumulado'
    }
    
    # features onde valor BAIXO = ruim (ex.: lubrificação)
    baixo_ruim = {'lubrificacao', 'nivel_oleo', 'eficiencia'}

    nome_feat = feat.lower()
    is_baixo_ruim = any(k in nome_feat for k in ['lubri', 'oleo', 'efici'])
    is_alto_ruim  = not is_baixo_ruim

    if is_alto_ruim:
        if valor >= p75:
            return "🔴 Alta severidade — desgaste severo / risco elevado"
        elif valor >= p25:
            return "🟡 Moderada — atenção recomendada"
        else:
            return "🟢 Normal — dentro do esperado"
    else:
        if valor <= p25:
            return "🔴 Alta severidade — nível crítico, intervenção necessária"
        elif valor <= p75:
            return "🟡 Moderada — monitorar de perto"
        else:
            return "🟢 Normal — dentro do esperado"

def previsao_interativa():
    """Opção 1: usuário informa os valores dos sensores e recebe análise completa."""
    print("\n" + "=" * 62)
    print("  🔧  PREVISÃO DE FALHA — INSERÇÃO MANUAL DE DADOS")
    print("=" * 62)
    print("  Informe os valores dos sensores abaixo.")
    print("  (Pressione ENTER para usar a mediana historica)\n")

    valores = {}
    # apenas as features originais (sem as engineered, que são derivadas)
    features_base = [f for f in FEATURES if f not in ('termica_pressao', 'desgaste_acumulado')]

    for feat in features_base:
        mediana = df[feat].median()
        while True:
            entrada = input(f"  {feat:<22} [mediana={mediana:.2f}]: ").strip()
            if entrada == "":
                valores[feat] = mediana
                break
            try:
                valores[feat] = float(entrada)
                break
            except ValueError:
                print("  ⚠️  Valor inválido. Digite um número ou pressione ENTER.")

    # recalcula as features derivadas
    valores['termica_pressao']    = valores['temperatura'] * valores['pressao']
    valores['desgaste_acumulado'] = valores['vibracao']    * valores['tempo_operacao']

    entrada_df = pd.DataFrame([valores])[FEATURES]

    # previsão do modelo
    prob_falha  = modelo_rf.predict_proba(entrada_df)[0]
    classe      = modelo_rf.predict(entrada_df)[0]
    prob_falha_pct = prob_falha[1] * 100 if len(prob_falha) > 1 else prob_falha[0] * 100

    print("\n" + "=" * 62)
    print("  📊  ANÁLISE DE SEVERIDADE POR VARIÁVEL")
    print("=" * 62)

    for feat, val in valores.items():
        sev = severidade(val, feat)
        print(f"  {feat:<22}  val={val:>10.2f}  →  {sev}")

    # risco combinado temperatura + pressão
    lim_temp  = PERCENTIS['temperatura'][0.75]
    lim_press = PERCENTIS['pressao'][0.75]
    if valores['temperatura'] > lim_temp and valores['pressao'] > lim_press:
        print("\n  ⚠️  ALERTA COMBINADO: Temperatura + Pressão simultaneamente elevadas!")
        print("     Risco de dano térmico e ruptura. Parada imediata recomendada.")

    lim_vib = PERCENTIS['vibracao'][0.75]
    lim_top = PERCENTIS['tempo_operacao'][0.75]
    if valores['vibracao'] > lim_vib and valores['tempo_operacao'] > lim_top:
        print("\n  ⚠️  ALERTA COMBINADO: Vibração + Tempo de operação elevados!")
        print("     Desgaste acelerado de rolamentos. Agendar manutenção urgente.")

    print("\n" + "=" * 62)
    print("  🤖  PREVISÃO DO MODELO (Random Forest)")
    print("=" * 62)

    if classe == 1:
        status = "🔴 FALHA PREVISTA"
    else:
        if prob_falha_pct >= 30:
            status = "🟡 SEM FALHA — mas risco moderado"
        else:
            status = "🟢 SEM FALHA PREVISTA"

    print(f"  Resultado    : {status}")
    print(f"  Probabilidade de falha : {prob_falha_pct:.1f}%")
    print("=" * 62)


# menu principal

def menu():
    while True:
        print("\n" + "=" * 62)
        print("  🏭  SISTEMA DE MANUTENÇÃO INDUSTRIAL")
        print("=" * 62)
        print("  1. Fazer previsão (inserir dados dos sensores)")
        print("  2. Consultar a IA sobre os dados  (requer GOOGLE_API_KEY)")
        print("  3. Gerar relatório completo pela IA  (requer GOOGLE_API_KEY)")
        print("  4. Sair")
        print("=" * 62)

        escolha = input("\n  Escolha uma opcao (1-4): ").strip()

        # ------------------------------------------------------------------
        if escolha == '1':
            previsao_interativa()

        # ------------------------------------------------------------------
        elif escolha == '2':
            if not gemini_client:
                print("\n  ❌ IA não disponível. Configure GOOGLE_API_KEY no .env")
                continue

            pergunta = input("\n  O que você quer saber sobre os dados da máquina?\n  > ").strip()
            if not pergunta:
                continue

            print("\n  ⏳ Consultando o Gemini...")
            stats_resumo = df.describe().to_dict()
            falhas_resumo = {
                "total_registros"  : len(df),
                "total_falhas"     : int(df[TARGET].sum()),
                "taxa_falha_pct"   : round(df[TARGET].mean() * 100, 2),
                "top3_fatores"     : [f for f, _ in
                                       sorted(IMPORTANCIAS.items(),
                                              key=lambda x: x[1], reverse=True)[:3]]
            }

            prompt = f"""
Você é um especialista em manutenção industrial com 20 anos de experiência.
Abaixo estão as estatísticas dos sensores de uma máquina industrial coletadas
ao longo do tempo, além de um resumo dos fatores que mais influenciam falhas
segundo um modelo de Machine Learning (Random Forest).

== RESUMO DE FALHAS ==
{falhas_resumo}

== ESTATÍSTICAS DOS SENSORES ==
{stats_resumo}

Responda à seguinte pergunta do técnico de manutenção de forma clara, objetiva
e em português:

{pergunta}
"""
            try:
                resposta = gemini_client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt
                )
                print("\n  💡 RESPOSTA:")
                print("  " + "-" * 58)
                for linha in resposta.text.splitlines():
                    print(f"  {linha}")
                print("  " + "-" * 58)
            except Exception as e:
                print(f"\n  ❌ Erro ao consultar o Gemini: {e}")

        # ------------------------------------------------------------------
        elif escolha == '3':
            if not gemini_client:
                print("\n  ❌ IA não disponível. Configure GOOGLE_API_KEY no .env")
                continue

            print("\n  ⏳ Gerando relatório completo com IA... Aguarde.")

            importancias_texto = "\n".join(
                [f"  - {feat}: {imp:.4f}" for feat, imp
                 in sorted(IMPORTANCIAS.items(), key=lambda x: x[1], reverse=True)]
            )

            prompt = f"""
Você é um engenheiro sênior de manutenção industrial. Crie um RELATÓRIO TÉCNICO
COMPLETO E DETALHADO em português, com base nos dados abaixo.

O relatório deve conter obrigatoriamente as seguintes seções:
1. Sumário Executivo
2. Diagnóstico Geral da Máquina
3. Análise dos Sensores (quais variam mais, quais são críticos)
4. Fatores de Maior Risco de Falha (com base nas importâncias do modelo)
5. Situações de Risco Combinado (ex.: pressão alta + temperatura alta)
6. Recomendações de Manutenção Preventiva e Corretiva
7. Cronograma sugerido de inspeções

== ESTATÍSTICAS DOS SENSORES ==
{df.describe().to_dict()}

== IMPORTÂNCIA DAS VARIÁVEIS NO MODELO ==
{importancias_texto}

== RESUMO DE FALHAS ==
Total de registros : {len(df)}
Total de falhas    : {int(df[TARGET].sum())}
Taxa de falha      : {df[TARGET].mean()*100:.2f}%
Acurácia do modelo : {acuracia:.2%}

Escreva o relatório de forma formal, técnica e bem estruturada.
"""
            try:
                resposta = gemini_client.models.generate_content(
                    model=GEMINI_MODEL, contents=prompt
                )
                nome_arquivo = f"relatorio_manutencao_ia_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                with open(nome_arquivo, "w", encoding="utf-8") as arq:
                    arq.write(resposta.text)
                print(f"\n  ✅ Relatório salvo em: '{nome_arquivo}'")
                print("\n  📄 Prévia do relatório:")
                print("  " + "-" * 58)
                for linha in resposta.text.splitlines()[:20]:
                    print(f"  {linha}")
                print("  ... (veja o arquivo completo para o relatório inteiro)")
                print("  " + "-" * 58)
            except Exception as e:
                print(f"\n  ❌ Erro ao gerar o relatório: {e}")

        # ------------------------------------------------------------------
        elif escolha == '4':
            print("\n  👋 Saindo do sistema. Ate logo!\n")
            sys.exit(0)

        else:
            print("\n  ❌ Opção inválida. Digite 1, 2, 3 ou 4.")


# ponto de entrada

if __name__ == "__main__":
    gerar_relatorio_tecnico()   # relatório técnico sem IA ao iniciar
    menu()                      # abre o menu interativo