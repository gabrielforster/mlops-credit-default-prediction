import pytest

from src.demo import EXEMPLO, avaliar_cliente, faixa_de_risco
from src.utils import load_config, project_root, resolve_path

ROOT = project_root()
pytestmark = pytest.mark.skipif(not resolve_path(load_config(ROOT), "manifest", ROOT).is_file(),
                                reason="requer um modelo vigente versionado")


def test_faixa_de_risco_limites():
    assert [faixa_de_risco(s) for s in (0.0, 0.3, 0.5, 0.7)] == ["baixo", "moderado", "alto", "muito alto"]


def test_avaliar_cliente_explica_cada_atributo_do_modelo():
    r = avaliar_cliente(EXEMPLO, ROOT)
    assert 0.0 <= r["pontuacao"] <= 1.0
    assert r["classe"] == int(r["pontuacao"] >= 0.5)
    assert sorted(f["atributo"] for f in r["fatores"]) == sorted(r["atributos"])
    efeitos = [abs(f["efeito"]) for f in r["fatores"]]
    assert efeitos == sorted(efeitos, reverse=True)


def test_avaliar_cliente_rejeita_entrada_incompleta_ou_fora_do_dominio():
    with pytest.raises(ValueError, match="ausentes"):
        avaliar_cliente({"LIMIT_BAL": 1000}, ROOT)
    with pytest.raises(ValueError, match="AGE"):
        avaliar_cliente({**EXEMPLO, "AGE": 7}, ROOT)
