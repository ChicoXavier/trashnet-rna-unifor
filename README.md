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
│   ├── artigo_abnt.tex                      # Fonte LaTeX espelhando o artigo ABNT
│   └── Artigo_TrashNet_RNA_Unifor_ABNT.pdf  # Artigo técnico final pronto para envio no AVA (9 páginas)
├── data/
│   ├── dataset-resized.zip                  # Dataset oficial TrashNet (2.527 imagens originais em 512 × 384)
│   ├── dataset-resized/                     # Imagens extraídas por classe (6 duplicadas conflitantes excluídas no split)
│   └── splits/                              # Divisão estratificada fixa (seed = 42, 2.521 imagens limpas)
│       ├── dataset_split_seed42.csv
│       ├── train_seed42.csv                 # 1.764 imagens (70%)
│       ├── val_seed42.csv                   #   378 imagens (15%)
│       └── test_seed42.csv                  #   379 imagens (15%)
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

Utiliza-se exclusivamente o arquivo oficial `dataset-resized.zip` do repositório **TrashNet** (2.527 imagens originais em `512 × 384` pixels). Após a remoção de **6 arquivos** referentes a 3 pares de imagens idênticas (mesmo MD5) rotulados em classes conflitantes (`plastic152.jpg`/`glass176.jpg`, `plastic332.jpg`/`glass389.jpg`, `metal91.jpg`/`glass115.jpg`), restaram **2.521 imagens** válidas:
- `glass`: 498 | `paper`: 594 | `cardboard`: 403 | `plastic`: 480 | `metal`: 409 | `trash`: 137

A divisão estratificada fixa (`seed = 42`) está registrada em `data/splits/` (`70%` treino = `1.764`, `15%` validação = `378`, `15%` teste = `379`) e é compartilhada por todos os seis experimentos obrigatórios e pelo experimento bônus com YOLO.

---

## 4. Avaliação Sem Novo Treinamento (Reprodutibilidade)

Com o ambiente virtual ativo (`source .venv/bin/activate`), execute o comando abaixo para validar imediatamente as métricas no conjunto de teste (`379` imagens) carregando diretamente os modelos salvos em `models/` **sem treinar novamente**:

```bash
python src/evaluate_saved_models.py
```

---

## 5. Como Reproduzir o Pipeline Completo do Zero (Opcional)

1. **Análise Exploratória e Divisão Fixa**:
   ```bash
   python src/eda_and_split.py
   ```
2. **Treinamento dos 6 Experimentos Obrigatórios**:
   ```bash
   python src/train_experiments.py
   ```
3. **Treinamento do Objetivo Extra (YOLO11n-cls)**:
   ```bash
   python src/train_yolo_bonus.py
   ```
4. **Geração de Gráficos e Tabelas**:
   ```bash
   python src/generate_figures_and_tables.py
   ```

---

## 6. Resumo dos Resultados Obtidos no Teste (2.521 imagens limpas)

| Experimento | Resolução (L×A) | Pesos de Classe | Parâmetros | Tempo Total (s) | Inferência (ms/img) | Teste Acc | Teste F1 Macro |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Exp 1: MLP | `128 × 96` | Não | `9.471.110` | `70,97` | `0,24` | `34,83%` | `0,2275` |
| Exp 2: MLP | `128 × 96` | Sim | `9.471.110` | `64,91` | `0,23` | `25,86%` | `0,1100` |
| Exp 3: CNN | `128 × 96` | Não | `28.358` | `213,74` | `0,62` | **`74,67%`** | `0,7154` |
| **Exp 4: CNN (Melhor Val)** | `128 × 96` | Sim | `28.358` | `200,80` | `0,82` | `73,09%` | `0,7036` |
| Exp 5: CNN | `256 × 192` | Não | `28.358` | `1.206,65` | `3,74` | **`74,67%`** | **`0,7213`** |
| Exp 6: CNN | `512 × 384` | Não | `28.358` | `3.088,45` | `18,07` | `70,98%` | `0,6582` |
| **Exp 7 (Bônus): YOLO11n-cls** | `224 × 224` | Pré-treinado | `1.538.790` | `2.462,58` | `16,70` | **`91,03%`** | **`0,8865`** |
