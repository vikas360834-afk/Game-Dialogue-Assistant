from app.storage import Store
from app import llm

def test_store_roundtrip_and_append(tmp_path):
    s=Store(tmp_path/"data.json")
    s.put("profile",{"name":"Mira"})
    s.append("history","hello")
    assert s.get("profile")["name"]=="Mira"
    assert s.get("history")==["hello"]
    s.delete("profile")
    assert s.get("profile") is None

def test_llm_settings_default(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    for k in ("LLM_BASE_URL","LLM_API_KEY","LLM_MODEL"): monkeypatch.delenv(k,raising=False)
    assert llm.settings()==("http://localhost:11434/v1","","")

def test_cli_commands(capsys):
    from app.cli import main
    main(["hello"])
    assert "ready" in capsys.readouterr().out
    main(["train"])
    assert "Vocabulary:" in capsys.readouterr().out
    main(["generate","The kingdom","--length","8","--options","2","--engine","ngram","--seed","4"])
    output=capsys.readouterr().out
    assert "Option 1:" in output and "Option 2:" in output
    main(["evaluate"])
    assert "perplexity" in capsys.readouterr().out.lower()
