"""Command-line interface."""
import argparse, json
from pathlib import Path
from .core import DEFAULT_CORPUS, NGramModel, load_corpus, split_tokens, generate_dialogue
from .llm import hybrid_generate
from .storage import Store

def build_parser():
    p=argparse.ArgumentParser(prog="game-dialogue-assistant",description="Hybrid NPC dialogue studio")
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("hello")
    t=sub.add_parser("train"); t.add_argument("--corpus",type=Path,default=DEFAULT_CORPUS); t.add_argument("--n",type=int,default=3)
    g=sub.add_parser("generate"); g.add_argument("prompt",nargs="?",default="")
    g.add_argument("--length",type=int,default=40); g.add_argument("--options",type=int,default=1)
    g.add_argument("--n",type=int,default=3); g.add_argument("--corpus",type=Path,default=DEFAULT_CORPUS)
    g.add_argument("--seed",type=int); g.add_argument("--temperature",type=float,default=.85)
    g.add_argument("--personality",choices=["warrior","merchant","wizard","villager","villain","royal"],default="villager")
    g.add_argument("--emotion",choices=["neutral","happy","angry","sad","afraid","sarcastic","mysterious"],default="neutral")
    g.add_argument("--location",default="village"); g.add_argument("--context",default="")
    g.add_argument("--engine",choices=["ngram","hybrid","llm"],default="hybrid")
    e=sub.add_parser("evaluate"); e.add_argument("--corpus",type=Path,default=DEFAULT_CORPUS); e.add_argument("--n",type=int,default=3); e.add_argument("--train-ratio",type=float,default=.8)
    i=sub.add_parser("interactive"); i.add_argument("--length",type=int,default=40); i.add_argument("--n",type=int,default=3); i.add_argument("--corpus",type=Path,default=DEFAULT_CORPUS)
    s=sub.add_parser("save"); s.add_argument("name"); s.add_argument("text",nargs="+")
    ex=sub.add_parser("export"); ex.add_argument("--format",choices=["json","txt"],default="json"); ex.add_argument("--output",default="dialogue-export")
    return p

def main(argv=None):
    a=build_parser().parse_args(argv)
    if a.command=="hello":
        print("Game Dialogue Assistant 2.0 is ready. Try: python -m app.cli generate 'The kingdom' --personality wizard"); return
    store=Store()
    if a.command=="save":
        store.append("saved_dialogues",{"name":a.name,"text":" ".join(a.text)})
        print(f"Saved dialogue: {a.name}"); return
    if a.command=="export":
        data=store.get("saved_dialogues",[])
        path=Path(a.output).with_suffix(".json" if a.format=="json" else ".txt")
        path.parent.mkdir(parents=True,exist_ok=True)
        if a.format=="json": path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding="utf-8")
        else: path.write_text("\n\n".join(f"{x['name']}\n{x['text']}" for x in data),encoding="utf-8")
        print(f"Exported {len(data)} dialogue(s) to {path}"); return
    tokens=load_corpus(a.corpus)
    if a.command=="evaluate":
        train,test=split_tokens(tokens,a.train_ratio); model=NGramModel(a.n).fit(train)
        print(f"Training tokens: {len(train)}\nHeld-out tokens: {len(test)}\n{a.n}-gram perplexity: {model.evaluate_perplexity(test):.3f}")
        return
    model=NGramModel(a.n).fit(tokens)
    if a.command=="train":
        print(f"Corpus: {a.corpus}\nTokens: {len(tokens)}\nVocabulary: {len(model.vocabulary)}\nN-gram order: {a.n}\nSmoothing: Add-1"); return
    if a.command=="interactive":
        history=[]
        print("Interactive NPC dialogue. Type /quit to exit; /clear clears history.")
        while True:
            try: prompt=input("\nYou> ").strip()
            except (EOFError,KeyboardInterrupt): print("\nGoodbye!"); break
            if prompt=="/quit": break
            if prompt=="/clear": history=[]; print("History cleared."); continue
            if not prompt: continue
            reply=hybrid_generate(model,prompt,history=history,length=a.length)
            print("NPC>",reply); history.extend([prompt,reply])
        return
    if a.command=="generate":
        if a.length<1 or a.options<1: raise SystemExit("--length and --options must be positive")
        for idx in range(a.options):
            if a.engine=="ngram":
                result=generate_dialogue(model,a.prompt,a.personality,a.emotion,a.location,a.context,length=a.length,
                                         temperature=a.temperature,seed=None if a.seed is None else a.seed+idx)[0]
            else:
                result=hybrid_generate(model,a.prompt,a.personality,a.emotion,a.location,a.context,
                                       length=a.length,engine=a.engine,temperature=a.temperature)
            print(f"Option {idx+1}: {result}")

if __name__=="__main__": main()
