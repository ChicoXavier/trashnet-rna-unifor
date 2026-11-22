#!/usr/bin/env python3
"""
Gera os três notebooks Jupyter exigidos na Seção 6 do repositório GitHub:
  1. notebooks/01_analise_exploratoria.ipynb
  2. notebooks/02_treinamento_experimentos.ipynb
  3. notebooks/03_resultados_e_analise.ipynb
"""

from pathlib import Path
import nbformat as nbf

BASE_DIR = Path(__file__).resolve().parent.parent
NB_DIR = BASE_DIR / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def make_nb(cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return nb


def build_nb1_eda():
    cells = [
        nbf.v4.new_markdown_cell(
            "# 01 — Análise Exploratória de Dados (EDA) e Divisão Estratificada\n\n"
            "**Atividade Final de Redes Neurais Artificiais — EAD Unifor (TrashNet)**\n\n"
            "Este notebook inspeciona o dataset oficial `dataset-resized.zip` (2.527 imagens em `512 × 384`), "
            "verifica integridade, dimensões, canais de cor e duplicações, realiza a divisão estratificada fixa "
            "(70% treino, 15% validação, 15% teste com `seed = 42`) e exibe a distribuição e exemplos das 6 classes."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "BASE_DIR = Path('..').resolve()\n"
            "with open(BASE_DIR / 'results' / 'eda_summary.json', 'r', encoding='utf-8') as f:\n"
            "    eda = json.load(f)\n\n"
            "print('Total de imagens:', eda['total_images'])\n"
            "print('Arquivos corrompidos:', eda['corrupted_files'])\n"
            "print('Dimensões:', eda['dimensions_summary'])\n"
            "print('Modos de cor:', eda['color_modes'])\n"
            "print('Pares MD5 idênticos encontrados:', eda['duplicate_md5_pairs'])\n"
            "print('Totais por divisão (seed=42):', eda['split_totals'])\n"
            "print('Média RGB no treino [0,1]:', eda['train_rgb_mean_0_1'])\n"
            "print('Desvio padrão RGB no treino [0,1]:', eda['train_rgb_std_0_1'])"
        ),
        nbf.v4.new_code_cell(
            "df_dist = pd.DataFrame(eda['class_distribution']).T\n"
            "display(df_dist)"
        ),
        nbf.v4.new_markdown_cell("## Distribuição Estratificada e Exemplos Visuais"),
        nbf.v4.new_code_cell(
            "for fig_name in ['fig1_distribuicao_classes.png', 'fig2_exemplos_classes.png', 'fig3_comparacao_resolucoes.png']:\n"
            "    display(Image(filename=str(BASE_DIR / 'results' / 'figures' / fig_name)))"
        ),
    ]
    nbf.write(make_nb(cells), NB_DIR / "01_analise_exploratoria.ipynb")


def build_nb2_training():
    cells = [
        nbf.v4.new_markdown_cell(
            "# 02 — Protocolo de Treinamento dos 6 Experimentos Obrigatórios + Bônus YOLO\n\n"
            "Este notebook documenta as arquiteturas **MLP** e **CNN** implementadas em PyTorch "
            "([`src/train_experiments.py`](../src/train_experiments.py)), o esquema de ajuste fino do **YOLO11n-cls** "
            "([`src/train_yolo_bonus.py`](../src/train_yolo_bonus.py)) e o comando de execução."
        ),
        nbf.v4.new_code_cell(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.append(str(Path('..').resolve() / 'src'))\n\n"
            "from train_experiments import TrashNetMLP, TrashNetCNN, count_parameters\n\n"
            "mlp = TrashNetMLP(height=96, width=128, num_classes=6)\n"
            "cnn = TrashNetCNN(num_classes=6)\n"
            "print(f'Parâmetros MLP (128x96): {count_parameters(mlp):,}')\n"
            "print(f'Parâmetros CNN (128x96, 256x192 e 512x384 com GlobalAvgPool): {count_parameters(cnn):,}')\n"
            "print('\\nResumo MLP:\\n', mlp)\n"
            "print('\\nResumo CNN:\\n', cnn)"
        ),
        nbf.v4.new_markdown_cell(
            "### Como reproduzir o treinamento completo (opcional)\n"
            "```bash\n"
            ".venv/bin/python src/train_experiments.py\n"
            ".venv/bin/python src/train_yolo_bonus.py\n"
            "```"
        ),
    ]
    nbf.write(make_nb(cells), NB_DIR / "02_treinamento_experimentos.ipynb")


def build_nb3_results():
    cells = [
        nbf.v4.new_markdown_cell(
            "# 03 — Resultados, Métricas, Gráficos e Avaliação Sem Novo Treinamento\n\n"
            "Este notebook apresenta todas as tabelas comparativas, métricas por classe, curvas de aprendizagem, "
            "matrizes de confusão, efeito da resolução na CNN, análise de erros do melhor modelo obrigatório e "
            "avaliação reprodutível direta dos checkpoints salvos em `models/`."
        ),
        nbf.v4.new_code_cell(
            "import json\n"
            "import sys\n"
            "from pathlib import Path\n"
            "import pandas as pd\n"
            "from IPython.display import Image, display\n\n"
            "BASE_DIR = Path('..').resolve()\n"
            "sys.path.append(str(BASE_DIR / 'src'))\n\n"
            "df_comp = pd.read_csv(BASE_DIR / 'results' / 'tabela_comparativa_geral.csv')\n"
            "display(df_comp)"
        ),
        nbf.v4.new_code_cell(
            "df_cls = pd.read_csv(BASE_DIR / 'results' / 'tabela_metricas_por_classe.csv')\n"
            "display(df_cls)"
        ),
        nbf.v4.new_markdown_cell("## Avaliação Reprodutível dos Modelos Salvos (Sem Novo Treinamento)"),
        nbf.v4.new_code_cell(
            "from evaluate_saved_models import evaluate_pytorch_checkpoints\n"
            "evaluate_pytorch_checkpoints()"
        ),
        nbf.v4.new_markdown_cell("## Painéis Gráficos Obrigatórios e Extensão YOLO"),
        nbf.v4.new_code_cell(
            "figs = [\n"
            "    'fig1_distribuicao_classes.png',\n"
            "    'fig2_exemplos_classes.png',\n"
            "    'fig3_comparacao_resolucoes.png',\n"
            "    'fig4_curvas_aprendizagem_6exp.png',\n"
            "    'fig5_matrizes_confusao_6exp.png',\n"
            "    'fig6_efeito_resolucao_cnn.png',\n"
            "    'fig7_analise_erros_melhor_modelo.png',\n"
            "    'fig8_yolo_bonus_avaliacao.png',\n"
            "]\n"
            "for fig in figs:\n"
            "    print('Exibindo:', fig)\n"
            "    display(Image(filename=str(BASE_DIR / 'results' / 'figures' / fig)))"
        ),
    ]
    nbf.write(make_nb(cells), NB_DIR / "03_resultados_e_analise.ipynb")


if __name__ == "__main__":
    build_nb1_eda()
    build_nb2_training()
    build_nb3_results()
    print("Notebooks gerados em notebooks/.")
