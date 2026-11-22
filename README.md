# Classificação de Resíduos por Imagens com Redes Neurais Artificiais (TrashNet)

Estudo experimental comparando **Multilayer Perceptron (MLP)**, **Redes Neurais Convolucionais (CNN)** em múltiplas resoluções e **Transfer Learning com YOLO11n-cls** (objetivo extra) na classificação das seis classes de resíduos do dataset **TrashNet** (`glass`, `paper`, `cardboard`, `plastic`, `metal`, `trash`).

Desenvolvido para a disciplina **Redes Neurais Artificiais** — **EAD Unifor (2026)**.

---

## 1. Estrutura do Repositório

```text
trashnet-rna-unifor/
├── README.md
├── requirements.txt
├── article/
│   ├── artigo_abnt.html                     # Fonte HTML/CSS formatada no padrão ABNT (1 coluna, <= 10 páginas)
│   └── Artigo_TrashNet_RNA_Unifor_ABNT.pdf  # Artigo técnico final pronto para envio no AVA
├── data/
│   ├── dataset-resized.zip                  # Dataset oficial TrashNet (2.527 imagens em 512 × 384)
│   ├── dataset-resized/                     # Imagens extraídas por classe
│   └── splits/                              # Divisão estratificada fixa (seed = 42)
│       ├── dataset_split_seed42.csv
│       ├── train_seed42.csv                 # 1.768 imagens (70%)
│       ├── val_seed42.csv                   #   379 imagens (15%)
│       └── test_seed42.csv                  #   380 imagens (15%)
├── models/                                  # Checkpoints salvos (melhores pesos na validação)
│   ├── exp1_mlp_128x96_sem_pesos.pt         # 36,13 MB
│   ├── exp2_mlp_128x96_com_pesos.pt         # 36,13 MB
│   ├── exp3_cnn_128x96_sem_pesos.pt         #  0,12 MB
│   ├── exp4_cnn_128x96_com_pesos.pt         #  0,12 MB
│   ├── exp5_cnn_256x192_sem_pesos.pt        #  0,12 MB
│   ├── exp6_cnn_512x384_sem_pesos.pt        #  0,12 MB
│   └── exp7_yolo11n_cls_bonus.pt            #  3,05 MB (Objetivo Extra YOLO)
├── notebooks/
│   ├── 01_analise_exploratoria.ipynb        # Inspeção, distribuição, exemplos e pré-processamento
│   ├── 02_treinamento_experimentos.ipynb    # Arquiteturas MLP/CNN e protocolo controlado
│   └── 03_resultados_e_analise.ipynb        # Métricas, matrizes, curvas, erros e avaliação reprodutível
├── results/
│   ├── eda_summary.json                     # Estatísticas da EDA e pesos de classe
│   ├── experiments_results.json             # Resultados e históricos dos 6 experimentos obrigatórios
│   ├── yolo_bonus_results.json              # Resultados do objetivo extra YOLO11n-cls
│   ├── tabela_comparativa_geral.csv         # Métricas e custo computacional consolidados
│   ├── tabela_metricas_por_classe.csv       # Precisão, recall, F1 e suporte por classe
│   ├── history_*.csv                        # Históricos por época de cada experimento
│   ├── predictions_*.csv                    # Previsões individuais no conjunto de teste
│   └── figures/                             # Painéis e gráficos exigidos (fig1 a fig8)
└── src/
    ├── eda_and_split.py                     # Etapa 1: EDA, divisão estratificada fixa e figuras iniciais
    ├── train_experiments.py                 # Etapa 2: Treinamento dos 6 experimentos obrigatórios
    ├── train_yolo_bonus.py                  # Etapa 2b: Ajuste fino do YOLO11n-cls (Seção 7)
    ├── generate_figures_and_tables.py       # Etapa 3: Curvas, matrizes, resolução, erros e tabelas
    ├── evaluate_saved_models.py             # Avaliação reprodutível SEM novo treinamento
    └── build_notebooks.py                   # Geração programática dos notebooks
```

---

## 2. Dependências e Instalação

- **Python**: `3.11` / `3.12`
- **Principais bibliotecas**:
  - `torch` (`2.2.2` no treinamento em macOS / `2.15.0` na avaliação Linux)
  - `torchvision` (`0.17.2` / `0.29.2`)
  - `ultralytics` (`8.4.216`)
  - `scikit-learn` (`1.9.1`)
  - `numpy` (`1.26.4` / `2.5.5`), `pandas`, `pillow`, `matplotlib`, `seaborn`, `psutil`, `jupyter`

### Instalação do ambiente e download dos dados

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Baixar e extrair o dataset oficial TrashNet (se ainda não estiver em data/dataset-resized/)
mkdir -p data && curl -L -o data/dataset-resized.zip "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip" && unzip -q data/dataset-resized.zip -d data/
```

---

## 3. Dataset e Divisão Fixa

Utiliza-se exclusivamente o arquivo oficial `dataset-resized.zip` do repositório **TrashNet** (2.527 imagens RGB em `512 × 384` pixels):
- `glass`: 501 | `paper`: 594 | `cardboard`: 403 | `plastic`: 482 | `metal`: 410 | `trash`: 137

A divisão estratificada fixa (`seed = 42`) está registrada em `data/splits/` (`70%` treino = `1.768`, `15%` validação = `379`, `15%` teste = `380`) e é compartilhada por todos os seis experimentos obrigatórios e pelo experimento bônus com YOLO.

---

## 4. Avaliação Sem Novo Treinamento (Reprodutibilidade)

Para validar imediatamente as métricas no conjunto de teste (`380` imagens) carregando diretamente os modelos salvos em `models/` **sem treinar novamente**:

```bash
.venv/bin/python src/evaluate_saved_models.py
```

---

## 5. Como Reproduzir o Pipeline Completo do Zero (Opcional)

1. **Análise Exploratória e Divisão Fixa**:
   ```bash
   .venv/bin/python src/eda_and_split.py
   ```
2. **Treinamento dos 6 Experimentos Obrigatórios**:
   ```bash
   .venv/bin/python src/train_experiments.py
   ```
3. **Treinamento do Objetivo Extra (YOLO11n-cls)**:
   ```bash
   .venv/bin/python src/train_yolo_bonus.py
   ```
4. **Geração de Gráficos e Tabelas**:
   ```bash
   .venv/bin/python src/generate_figures_and_tables.py
   ```

---

## 6. Resumo dos Resultados Obtidos no Teste

| Experimento | Resolução (L×A) | Pesos de Classe | Parâmetros | Tempo Total (s) | Teste Acc | Teste F1 Macro |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Exp 1: MLP | `128 × 96` | Não | `9.471.110` | `62,04` | `38,42%` | `0,2941` |
| Exp 2: MLP | `128 × 96` | Sim | `9.471.110` | `43,40` | `28,95%` | `0,2005` |
| **Exp 3: CNN** | `128 × 96` | Não | `28.358` | `102,70` | **`72,37%`** | `0,6754` |
| **Exp 4: CNN** | `128 × 96` | Sim | `28.358` | `101,11` | `70,53%` | **`0,6833`** |
| Exp 5: CNN | `256 × 192` | Não | `28.358` | `340,62` | `71,84%` | `0,6694` |
| Exp 6: CNN | `512 × 384` | Não | `28.358` | `708,35` | `56,84%` | `0,4992` |
| **Exp 7 (Bônus): YOLO11n-cls** | `224 × 224` | Pré-treinado | `1.538.790` | `2.404,90` | **`93,16%`** | **`0,9198`** |
