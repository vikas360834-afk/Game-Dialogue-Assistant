"""Optional Streamlit web dashboard. Run: streamlit run app/web.py"""
import json
from datetime import datetime
from pathlib import Path
import streamlit as st
from app.core import DEFAULT_CORPUS, NGramModel, load_corpus, generate_dialogue, PERSONALITIES, EMOTIONS
from app.llm import hybrid_generate, is_configured
from app.storage import Store

st.set_page_config(page_title="Game Dialogue Assistant",page_icon="🎮",layout="wide")
st.markdown("""<style>
.stApp{background:linear-gradient(135deg,#0b1020 0%,#111a32 55%,#17112b 100%);color:#eef3ff}
[data-testid="stMetric"],[data-testid="stVerticalBlockBorderWrapper"]{background:rgba(255,255,255,.055);border:1px solid rgba(160,190,255,.18);border-radius:16px;padding:12px}
.stButton>button{border-radius:12px;border:1px solid #6578b8;background:linear-gradient(120deg,#263b70,#4c3278);color:white}
</style>""",unsafe_allow_html=True)
store=Store()
@st.cache_resource
def get_model():
    return NGramModel(3).fit(load_corpus())
model=get_model()
st.title("🎮 Game Dialogue Assistant")
st.caption("NPC dialogue studio · Add-1 n-gram model + optional LLM enhancement")
with st.sidebar:
    st.header("Character setup")
    name=st.text_input("NPC name","Alden")
    personality=st.selectbox("Personality",list(PERSONALITIES))
    emotion=st.selectbox("Emotion",list(EMOTIONS))
    location=st.selectbox("Location",["village","castle","forest","dungeon","market","tavern","battlefield","ancient temple"])
    context=st.text_area("World / quest context","The northern gate has been sealed.")
    engine=st.selectbox("Generation engine",["hybrid","ngram","llm"])
    length=st.slider("Dialogue length (words/tokens approx.)",10,120,45)
    temperature=st.slider("Creativity",0.2,1.5,.85,.05)
    st.caption("LLM configured" if is_configured() else "LLM not configured — hybrid mode uses local n-gram fallback.")
tabs=st.tabs(["✍️ Generate","🧑‍🤝‍🧑 NPC Profiles","📚 Saved Dialogue","📊 Model Lab","📦 Import / Export"])
with tabs[0]:
    st.subheader("Create a conversation")
    prompt=st.text_input("Opening phrase","The kingdom is in danger")
    player=st.text_input("Optional player line","What happened here?")
    if st.button("Generate dialogue",type="primary"):
        history=st.session_state.get("history",[])
        full_prompt=(player+" "+prompt).strip()
        try:
            if engine=="ngram":
                outputs=generate_dialogue(model,full_prompt,personality,emotion,location,context,history,length,3,temperature)
            else:
                outputs=[hybrid_generate(model,full_prompt,personality,emotion,location,context,history,length,engine,temperature)]
            st.session_state["latest"]=outputs
            st.session_state["latest_meta"]={"npc":name,"personality":personality,"emotion":emotion,"location":location}
        except Exception as exc: st.error(str(exc))
    for i,dialogue in enumerate(st.session_state.get("latest",[])):
        with st.container(border=True):
            st.markdown(f"**{name} · Option {i+1}**")
            st.write(dialogue)
            if st.button("Save this dialogue",key=f"save{i}"):
                store.append("saved_dialogues",{"name":name,"text":dialogue,"personality":personality,
                    "emotion":emotion,"location":location,"created_at":datetime.now().isoformat()})
                st.success("Saved.")
    if st.session_state.get("latest"):
        if st.button("Add latest to conversation memory"):
            st.session_state.setdefault("history",[]).extend([prompt,*st.session_state["latest"]])
            st.success("Conversation memory updated.")
        if st.button("Clear conversation memory"):
            st.session_state["history"]=[]
with tabs[1]:
    st.subheader("NPC character profiles")
    profiles=store.get("npc_profiles",[])
    with st.form("npc_form"):
        npc_name=st.text_input("Character name")
        role=st.selectbox("Role",list(PERSONALITIES))
        traits=st.text_input("Traits","brave, loyal, secretive")
        backstory=st.text_area("Backstory")
        submitted=st.form_submit_button("Save NPC profile")
    if submitted and npc_name:
        profiles.append({"name":npc_name,"role":role,"traits":traits,"backstory":backstory})
        store.put("npc_profiles",profiles); st.success("Profile saved.")
    for profile in profiles: st.json(profile)
with tabs[2]:
    st.subheader("Dialogue library")
    saved=store.get("saved_dialogues",[])
    st.metric("Saved snippets",len(saved))
    for idx,item in enumerate(reversed(saved)):
        with st.expander(item.get("name","NPC dialogue")):
            st.write(item["text"]); st.caption(" · ".join(str(item.get(k,"")) for k in ["personality","emotion","location"]))
            if st.button("Delete",key=f"del{idx}"):
                saved.pop(len(saved)-1-idx); store.put("saved_dialogues",saved); st.rerun()
with tabs[3]:
    st.subheader("N-gram model diagnostics")
    c1,c2,c3=st.columns(3)
    c1.metric("Corpus tokens",model.total_tokens); c2.metric("Vocabulary",len(model.vocabulary)); c3.metric("N-gram order",model.n)
    st.write("This is a compact statistical language model. It is useful for demonstrating tokenization, conditional probabilities and sampling, but a small corpus can produce repetition or weak long-range coherence.")
    if st.button("Evaluate held-out perplexity"):
        tokens=load_corpus(); cut=int(len(tokens)*.8)
        if cut>2 and cut<len(tokens):
            eval_model=NGramModel(3).fit(tokens[:cut])
            st.metric("Held-out perplexity",f"{eval_model.evaluate_perplexity(tokens[cut:]):.2f}")
with tabs[4]:
    st.subheader("Bring your own game scripts")
    upload=st.file_uploader("Upload TXT dialogue corpus",type=["txt"])
    if upload and st.button("Save uploaded corpus"):
        target=Path("data/imported_dialogues.txt"); target.parent.mkdir(exist_ok=True)
        target.write_bytes(upload.getvalue()); st.success(f"Saved {target}. Use --corpus {target} in CLI; restart app to retrain.")
    saved=store.get("saved_dialogues",[])
    st.download_button("Export saved dialogue as JSON",json.dumps(saved,indent=2,ensure_ascii=False),
                       file_name="game-dialogues.json",mime="application/json")
    st.download_button("Export saved dialogue as TXT","\n\n".join(x["text"] for x in saved),
                       file_name="game-dialogues.txt",mime="text/plain")
