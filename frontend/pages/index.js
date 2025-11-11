import { useState } from "react";

export default function Home() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [loading, setLoading] = useState(false);

  const askAI = async () => {
    if (!question) return;
    setLoading(true);
    setAnswer("");
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    const data = await res.json();
    setAnswer(data.answer || "Error: No response.");
    setLoading(false);
  };

  return (
    <main style={{ padding: "2rem", fontFamily: "sans-serif" }}>
      <h1>🤖 AI Research Agent</h1>
      <textarea
        rows="4"
        placeholder="Ask your research question..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        style={{ width: "100%", marginTop: "1rem" }}
      />
      <button onClick={askAI} disabled={loading} style={{ marginTop: "1rem" }}>
        {loading ? "Thinking..." : "Ask"}
      </button>
      {answer && (
        <div style={{ marginTop: "2rem", whiteSpace: "pre-wrap" }}>
          <strong>Answer:</strong>
          <p>{answer}</p>
        </div>
      )}
    </main>
  );
}
