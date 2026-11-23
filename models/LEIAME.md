# Modelos

| Arquivo | O que é | Versionado na `main`? |
|---|---|---|
| `modelo_vigente.json` | Manifesto do modelo vigente e histórico de todas as decisões (identificação, `data_version`, features, métrica, `run_id`, decisão e motivo). Gerado pelo notebook 04 a cada ciclo. | Não. O oficial fica na branch `estado-pipeline`, gravado pelo workflow. |
| `modelo_vigente_actions.json` | Cópia do `modelo_vigente.json` da `estado-pipeline` | Sim (`python scripts/copiar_evidencias_actions.py`) |
| `registro/modelo_<cycle_id>_<A\|B>.joblib` | Vencedor de cada ciclo: `Pipeline` completo (pré-processamento, features eleitas e algoritmo) | Não (19 a 25 MB cada). No Actions, ficam na Release `modelos`. |
| `candidatos/<cycle_id>/` | Os dois candidatos do ciclo (notebook 03) | Não |

O `modelo_vigente.json` não é versionado porque cada execução local gera o seu. Se a `main` trouxesse o do Actions, o primeiro ciclo local procuraria um vigente que não está em `models/registro/` e pararia no notebook 04, que falha de propósito nesse caso para não promover sem comparar.

Para usar o vigente do Actions na demonstração (`python -m src.prever`), baixe o arquivo dele da Release e aponte o manifesto copiado:

```bash
arquivo=$(python -c "import json; print(json.load(open('models/modelo_vigente_actions.json'))['vigente']['arquivo'])")
gh release download modelos --pattern "$arquivo" --dir models/registro
python -m src.prever --json exemplos/reserva_exemplo.json --manifesto models/modelo_vigente_actions.json
```
