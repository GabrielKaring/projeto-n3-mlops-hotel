#!/usr/bin/env bash
# Executa um notebook do pipeline e grava a cópia executada em evidencias/execucao/ (F7).
#
# Uso: scripts/executar_notebook.sh <nome_sem_extensao>
#   ex.: scripts/executar_notebook.sh 03_treino_competicao
#
# O nbconvert roda sem a opção que tolera erros: um erro em qualquer célula faz o nbconvert sair com código != 0, o step falha e o
# workflow para ali, antes da promoção e da gravação do estado.
#
# FALHAR_EM=<nome_sem_extensao> simula uma falha antes desse notebook. Serve para demonstrar, no
# Actions, que uma falha interrompe o fluxo e impede a promoção (entrada "falhar_em" do workflow_dispatch).
set -euo pipefail

nome="${1:?informe o notebook, ex.: 01_etl}"
if [ ! -f "notebooks/${nome}.ipynb" ]; then
  echo "::error::notebooks/${nome}.ipynb não existe"
  exit 1
fi

if [ "${FALHAR_EM:-nenhum}" = "$nome" ]; then
  echo "::error::Falha simulada antes de ${nome} (FALHAR_EM=${FALHAR_EM}). O fluxo deve parar aqui."
  exit 1
fi

jupyter nbconvert --to notebook --execute "notebooks/${nome}.ipynb" \
  --output "${nome}_executado.ipynb" --output-dir evidencias/execucao \
  --ExecutePreprocessor.timeout=1800
