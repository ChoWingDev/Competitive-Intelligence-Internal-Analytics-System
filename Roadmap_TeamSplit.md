# SQL + Advanced RAG 雙驅動 PM 數據助手
## 四週全實作開發路線圖（2人協作版）

> **團隊分工原則：**
> - Person A（較強）→ 負責 Text-to-SQL、Router Agent、Parallel Merge 架構
> - Person B（較新）→ 負責 Advanced RAG Pipeline、UI Dashboard、Evaluation
> - 兩人都碰 LLM Prompting，都有完整的「我負責這塊」的面試故事
> - 每週末 sync：互相 code review，確保兩人都理解對方的模組

---

## 系統架構分工圖

```
PM 問題
    ↓
Business Glossary 層        ← 【A + B 共同設計】
    ↓
Router Agent                ← 【Person A 主責】
    ↓               ↓
Text-to-SQL       RAG Pipeline    ← 【A 主責 | B 主責】
公司 DB            行業報告
    ↓               ↓
    └── Merge + Insights 層 ──┘   ← 【Person A 主責】
                ↓
    PM Report UI + Evaluation     ← 【Person B 主責】
```

---

## Week 1：環境建設 + 各自啟動核心模組

**週目標：** 兩人並行啟動，Week 1 結束時各自有可跑的 Hello World。

### Person A 的任務（Text-to-SQL 基礎）

1. **環境建設（共同）**
   - 建立 GitHub repo，設定 branch 規範（`feature/sql-agent`, `feature/rag-pipeline`）
   - 建立 shared `requirements.txt` 和 `.env` 管理 API keys
   - 把 TheLook dataset CSV 導入 SQLite：
     ```python
     import sqlite3, pandas as pd
     tables = ['users', 'orders', 'order_items', 'products', 'events']
     conn = sqlite3.connect('thelook.db')
     for t in tables:
         df = pd.read_csv(f'data/{t}.csv')
         df.to_sql(t, conn, if_exists='replace', index=False)
     ```

2. **Business Glossary 設計（A+B 共同完成）**
   - 兩人一起定義 `glossary.json`，這是整個系統的地基：
     ```json
     {
       "churn_rate": "(lost_customers / total_customers_start_of_period) * 100",
       "ROI": "(sale_price - cost) / cost * 100",
       "new_campaign": "events WHERE traffic_source IS NOT NULL AND created_at >= first_day_of_last_month",
       "last_month": "DATE between first and last day of previous calendar month"
     }
     ```

3. **Text-to-SQL 基礎實作**
   - 用 LangChain `create_sql_agent` 連接 `thelook.db`
   - 讓 LLM 讀取 Schema，能回答：「上個月總收入是多少？」
   - 把 `glossary.json` 注入 System Prompt

4. **Week 1 交付物（A）：** 一個能跑的 SQL Agent，可以正確回答 3 條基本 PM 問題

### Person B 的任務（RAG Pipeline 基礎）

1. **環境建設（共同）** — 同上，與 A 一起完成

2. **Business Glossary 設計（A+B 共同完成）** — 同上

3. **文件準備 + 解析**
   - 下載 3–5 份電商行業報告（PDF）：
     - 推薦：Statista eCommerce reports、Shopify annual reports、行業 churn benchmark 報告
   - 用 `Unstructured` 解析 PDF，特別處理表格（轉 Markdown 格式）：
     ```python
     from unstructured.partition.pdf import partition_pdf
     elements = partition_pdf("report.pdf", strategy="hi_res")
     ```

4. **Chunking + Embedding**
   - 實作 `ParentDocumentRetriever`（大 chunk 儲存，小 chunk 檢索）
   - 建立 Chroma vector store，跑通基本語意搜尋

5. **Week 1 交付物（B）：** 一個能跑的 RAG chain，可以回答：「電商行業平均 churn rate 是多少？」

### 週末 Sync（必做）
- A 向 B 解釋 SQL Agent 怎麼讀 Schema、怎麼生成 SQL
- B 向 A 解釋 Chunking 策略、為什麼要用 ParentDocumentRetriever
- 目標：兩人都能向面試官解釋對方的模組

---

## Week 2：各自深化核心功能

**週目標：** 把各自的模組從「能跑」升級到「準確」。

### Person A 的任務（Text-to-SQL 深化）

1. **Few-Shot 提示詞優化**
   - 建立 PM 問題 → SQL 的範例庫，覆蓋 TheLook 的真實場景：
     ```python
     examples = [
       {
         "question": "上個月 organic search campaign 帶來的用戶，churn rate 是多少？",
         "sql": """
           WITH campaign_users AS (
             SELECT DISTINCT user_id FROM events
             WHERE traffic_source = 'Organic'
             AND DATE(created_at) >= DATE_TRUNC('month', DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH))
           ),
           churned AS (
             SELECT user_id FROM orders
             WHERE status = 'Cancelled' AND user_id IN (SELECT user_id FROM campaign_users)
           )
           SELECT COUNT(churned.user_id) * 100.0 / COUNT(campaign_users.user_id) as churn_rate
           FROM campaign_users LEFT JOIN churned USING (user_id)
         """
       }
     ]
     ```
   - 用 `FewShotPromptTemplate` 注入 LLM

2. **SQL 安全層**
   - Read-only 資料庫連接
   - SQL 審查過濾（拒絕 DROP、DELETE、UPDATE 等危險操作）：
     ```python
     def validate_sql(sql: str) -> bool:
         forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER']
         return not any(kw in sql.upper() for kw in forbidden)
     ```

3. **Python Analytics Sandbox**
   - 引入 `PythonREPLTool`，讓 SQL 結果可以進一步用 Pandas 分析
   - 生成 Matplotlib/Plotly 圖表並儲存為圖片

4. **Week 2 交付物（A）：** SQL Agent 能正確處理 multi-table JOIN，準確率達 7/10 測試問題

### Person B 的任務（RAG 深化）

1. **混合檢索（Hybrid Search）**
   - 結合 BM25（關鍵字）與 Vector Search（語意）：
     ```python
     from langchain.retrievers import BM25Retriever, EnsembleRetriever
     bm25 = BM25Retriever.from_documents(docs)
     vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
     hybrid = EnsembleRetriever(
         retrievers=[bm25, vector_retriever],
         weights=[0.4, 0.6]
     )
     ```

2. **RAG Prompt 優化**
   - 加入 citation 機制：讓 LLM 回答時標明來源文件和頁碼
   - 加入 hallucination guard：Prompt 明確要求「如果文件中沒有，請說不知道，不要猜測」

3. **行業數據專項處理**
   - 針對行業 benchmark 數據建立專門的 metadata 標籤（industry, year, metric_type）
   - 讓檢索時可以 filter：只找最近 2 年的電商 churn 數據

4. **Week 2 交付物（B）：** RAG 能準確回答 5 條行業 benchmark 問題，並附上來源引用

### 週末 Sync（必做）
- A + B 第一次做 integration test：手動把 SQL 結果和 RAG 結果放在一起，讓 LLM 嘗試對比
- 討論：兩個 output 格式差異在哪？Merge 層需要做什麼格式對齊？

---

## Week 3：整合 — Router、並行執行、記憶系統

**週目標：** 把兩個獨立模組連接成一個完整的系統。這週是整個專案技術含量最高的一週。

### Person A 的任務（Router + Parallel Merge）

1. **Router Agent**
   - 用 OpenAI Function Calling 定義三個 tool：
     ```python
     tools = [
         Tool(name="query_sql_db",
              description="用於查詢公司內部數據，如 campaign 表現、revenue、churn rate、ROI 等具體數字"),
         Tool(name="query_rag_documents",
              description="用於查詢行業報告、市場基準、競品數據等外部知識"),
         Tool(name="query_both",
              description="當問題同時需要公司數據和行業對比時使用，例如：我們的 churn rate 對比同行如何？")
     ]
     ```

2. **並行執行 + Merge 層（核心架構）**
   ```python
   import asyncio

   async def run_parallel_query(user_question: str, session_id: str):
       # 同時執行，節省時間
       sql_task = asyncio.create_task(sql_agent.arun(user_question))
       rag_task = asyncio.create_task(rag_chain.arun(user_question))
       sql_result, rag_result = await asyncio.gather(sql_task, rag_task)

       # Merge + Insights
       report = insights_chain.run({
           "your_data": sql_result,
           "industry_data": rag_result,
           "question": user_question
       })
       return report
   ```

3. **Insights Layer Prompt 設計**
   - 強制 JSON 輸出格式：
     ```
     你是一個資深電商數據分析師。
     公司數據：{your_data}
     行業基準：{industry_data}

     請生成 PM 報告，必須返回以下 JSON 格式：
     {
       "summary": "一句話總結",
       "comparison_table": [
         {"metric": "Churn Rate", "your_value": "X%", "industry_avg": "Y%", "status": "高於/低於行業"}
       ],
       "action_items": ["建議1", "建議2", "建議3"]
     }
     ```

4. **Week 3 交付物（A）：** 完整的 Router → Parallel Call → Merge 流程跑通，能生成 JSON 格式報告

### Person B 的任務（Memory + Session 管理）

1. **多輪對話記憶**
   - 實作 `ConversationBufferWindowMemory`（保留最近 5 輪對話）
   - 把 memory 注入 Router Agent，讓系統理解追問：
     - 第 1 句：「上個月 email campaign 的 ROI？」
     - 第 2 句：「那 organic campaign 呢？」← 系統要知道還是在問 ROI

2. **Session 持久化**
   - 用 SQLite 儲存對話紀錄（Production 概念示範）：
     ```python
     from langchain.memory import SQLChatMessageHistory
     history = SQLChatMessageHistory(
         session_id=session_id,
         connection_string="sqlite:///sessions.db"
     )
     ```

3. **Clarification Layer（加分項）**
   - 當問題太模糊時，系統主動追問而不是猜測：
     ```python
     def needs_clarification(question: str) -> bool:
         vague_keywords = ['表現', '怎樣', '好不好', 'performance']
         return any(kw in question for kw in vague_keywords)
     # 觸發時回應："你想了解哪個 campaign 的哪個指標？（churn rate / ROI / revenue）"
     ```

4. **Integration 協助**
   - 配合 A 的 Parallel Merge，確保 Memory 的 session_id 正確傳遞
   - 測試多輪對話在 Router 切換（SQL → RAG → SQL）時記憶是否保持

5. **Week 3 交付物（B）：** 多輪對話跑通，連續問 5 個追問問題系統都能理解上下文

### 週末 Sync（必做）
- Full end-to-end test：從 PM 問題 → Router → Parallel → Merge → JSON 報告
- 記錄失敗的問題，列成 Week 4 的修復清單

---

## Week 4：UI、評估、包裝

**週目標：** 把系統變成一個可以 Demo 的產品，並有數據證明它「真的準」。

### Person A 的任務（系統優化 + GitHub 包裝）

1. **效能調優**
   - Token 成本控制：Router 用 `gpt-4o-mini`，最終報告生成用 `gpt-4o`
   - 加入 caching：相同問題不重複打 API（用 `langchain.cache`）
   - 優化 SQL Agent 的 Schema 描述，減少 prompt token 用量

2. **Error Handling**
   - SQL 生成失敗時的 fallback 邏輯
   - RAG 找不到相關文件時的提示
   - JSON parse 失敗時的 retry 機制：
     ```python
     try:
         report = json.loads(llm_output)
     except json.JSONDecodeError:
         report = retry_with_format_correction(llm_output)
     ```

3. **GitHub README + 架構圖**
   - 寫精美的 `README.md`，包含：
     - 系統架構圖（可用 draw.io 或 Mermaid）
     - Demo GIF（錄製完整的 PM 問答流程）
     - 清晰的安裝步驟
     - 兩人的貢獻分工說明

4. **Week 4 交付物（A）：** 完整 GitHub repo，README 達到可以直接放 portfolio 的水準

### Person B 的任務（Streamlit UI + Evaluation）

1. **Streamlit Dashboard**
   - 對話框（Chat Interface）
   - 側邊欄：上傳新 PDF 報告、切換資料庫
   - 結構化報告渲染：
     ```python
     import streamlit as st
     import pandas as pd

     # 渲染 comparison table
     if 'comparison_table' in report:
         df = pd.DataFrame(report['comparison_table'])
         st.dataframe(df, use_container_width=True)

     # 渲染 action items
     if 'action_items' in report:
         st.subheader("行動建議")
         for i, item in enumerate(report['action_items'], 1):
             st.write(f"{i}. {item}")

     # 渲染圖表
     if chart_path:
         st.image(chart_path)
     ```

2. **RAG Evaluation（Eval 框架）**
   - 建立 20 條測試問題（10 SQL + 10 RAG），手動建立 Ground Truth
   - 用 RAG 三元組評估：
     ```python
     # 用 ragas 框架
     from ragas import evaluate
     from ragas.metrics import faithfulness, answer_relevancy, context_precision

     result = evaluate(
         dataset=test_dataset,
         metrics=[faithfulness, answer_relevancy, context_precision]
     )
     print(result)
     ```
   - 記錄準確率，寫進 README（這是面試的重要加分項）

3. **Demo 準備**
   - 準備 3 個「最好看」的 Demo 問題，確保 system 都能完美回答
   - 錄製 Demo GIF 給 A 放進 README

4. **Week 4 交付物（B）：** Streamlit app 跑通，Evaluation 報告顯示 RAG 準確率

### 週末 Sync（最終）
- 完整 Demo run-through，模擬面試官問「你負責哪個部分？你能解釋一下它怎麼運作？」
- 兩人互相練習對方模組的解釋

---

## 面試時各自怎麼說

### Person A 的面試話術
> "我負責整個系統的核心架構。我設計了 Router Agent，用 OpenAI Function Calling 讓 LLM 自主判斷要走 SQL 還是 RAG 還是兩者並行。最有挑戰性的部分是 Parallel Execution 加 Merge Layer — 我用 asyncio.gather 讓兩個 branch 同時執行，再設計了一個 Insights Chain 把公司數據和行業 benchmark 做對比分析，強制輸出 JSON 結構讓前端可以直接渲染成表格。"

### Person B 的面試話術
> "我負責 Advanced RAG Pipeline 和系統評估。我實作了 Hybrid Search，結合 BM25 和 Vector Search，解決了純語意搜尋在關鍵字查詢上的弱點。我也設計了整個 Evaluation 框架，用 Ragas 的 faithfulness 和 context precision 指標來量化系統的準確性，最終 RAG 的 faithfulness score 達到 X.XX。另外我做了 Streamlit Dashboard，把 LLM 的 JSON 輸出渲染成 PM 能直接使用的對比表格和圖表。"

---

## 整體分工總覽

| 模組 | 主責 | 協助 | 面試知識點 |
|---|---|---|---|
| Business Glossary | A + B | — | 業務語言 vs 技術定義的鴻溝 |
| TheLook DB 建設 | A | B | SQLite schema 設計 |
| Text-to-SQL Agent | A | — | Few-Shot prompting、SQL injection 防範 |
| Few-Shot 範例庫 | A | B review | Prompt engineering |
| PDF 解析 + Chunking | B | — | ParentDocumentRetriever、Overlap 策略 |
| Hybrid Search | B | — | BM25 vs Vector、Ensemble Retriever |
| Router Agent | A | — | Function Calling 底層原理 |
| Parallel Execution | A | — | asyncio、並行 vs 串行的時間差 |
| Merge + Insights Layer | A | B 測試 | Structured output、JSON schema |
| Session Memory | B | — | ConversationBufferWindowMemory、持久化 |
| Clarification Layer | B | — | 模糊問題處理策略 |
| Streamlit UI | B | — | 結構化輸出渲染 |
| Evaluation（Ragas） | B | — | RAG 三元組、LLM-as-a-judge |
| GitHub README + 架構圖 | A | B 提供 GIF | 系統設計能力展示 |
| Error Handling + Caching | A | — | Token 成本控制、Production 思維 |

---

## 每週 Sync Checklist

每週末兩人必須完成：
- [ ] Demo 對方負責的模組（不看 code，只看輸入輸出）
- [ ] 各自解釋本週最難的一個概念給對方聽
- [ ] Code review：每人至少 review 對方 1 個 PR
- [ ] 更新共享的測試問題庫（累積到 Week 4 的 20 條）
- [ ] 記錄 blockers，決定下週優先級
