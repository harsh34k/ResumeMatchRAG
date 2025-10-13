import React, { useState, useRef } from "react";
import axios from "axios";
import "./index.css";

interface Candidate {
  id: string;
  score: number;
  author: string;
  source: string;
  page: string;
  text_preview: string;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

function App() {
  const [files, setFiles] = useState<File[]>([]);
  const [jobDescription, setJobDescription] = useState("");
  const [topCandidates, setTopCandidates] = useState<Candidate[]>([]);
  const [sessionId, setSessionId] = useState(`session-${Date.now()}`);
  const [question, setQuestion] = useState("");
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleUpload = async () => {
    if (files.length === 0 || !jobDescription) return alert("Add files and job description");

    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));
    formData.append("job_description", jobDescription);

    try {
      setLoading(true);
      await axios.post("http://localhost:8000/upload_pdfs/", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      alert("Files uploaded successfully");
    } catch (err) {
      console.error(err);
      alert("Error uploading files");
    } finally {
      setLoading(false);
    }
  };

  const handleGetStarted = async () => {
    if (!jobDescription) return alert("Enter job description");
    const formData = new FormData();
    formData.append("job_description", jobDescription);

    try {
      setLoading(true);
      const res = await axios.post(
        "http://localhost:8000/ask/top_candidates/",
        formData
      );
      setTopCandidates(res.data.top_candidates || []);
    } catch (err) {
      console.error(err);
      alert("Error fetching top candidates");
    } finally {
      setLoading(false);
    }
  };


  const handleAskQuestion = async () => {
    if (!question) return;

    try {
      setLoading(true);
      const formData = new FormData();
      formData.append("session_id", sessionId);
      formData.append("question", question);

      const res = await axios.post("http://localhost:8000/ask/", formData);
      const answer = res.data.response || "";

      setChatHistory((prev) => [
        ...prev,
        { role: "user", content: question },
        { role: "assistant", content: answer },
      ]);
      setQuestion("");
    } catch (err) {
      console.error(err);
      alert("Error asking question");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen p-4 bg-gray-100">
      <h1 className="text-2xl font-bold mb-4">Resume Evaluator</h1>

      <div className="mb-4 p-4 bg-white rounded shadow">
        <input
          type="file"
          multiple
          ref={fileInputRef}
          onChange={(e) => setFiles(Array.from(e.target.files || []))}
          className="mb-2"
        />
        <textarea
          placeholder="Enter job description..."
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
          className="w-full border p-2 mb-2 rounded"
        />
        <button
          onClick={handleUpload}
          className="bg-blue-500 text-white px-4 py-2 rounded mr-2"
        >
          Upload Files
        </button>
        <button
          onClick={handleGetStarted}
          className="bg-green-500 text-white px-4 py-2 rounded"
        >
          Get Started
        </button>
      </div>

      {/* Top Candidates */}
      {topCandidates.length > 0 && (
        <div className="mb-4 p-4 bg-white rounded shadow">
          <h2 className="font-bold mb-2">Top Candidates</h2>
          {topCandidates.map((c) => (
            <div key={c.id} className="border p-2 mb-2 rounded">
              <p><strong>Author:</strong> {c.author}</p>
              <p><strong>Score:</strong> {c.score.toFixed(2)}</p>
              <p><strong>Source:</strong> {c.source}</p>
              <p><strong>Text Preview:</strong> {c.text_preview}</p>
            </div>
          ))}
        </div>
      )}

      {/* Chat Section */}
      <div className="p-4 bg-white rounded shadow">
        <h2 className="font-bold mb-2">Ask a Question</h2>
        <div className="flex flex-col mb-2">
          <textarea
            placeholder="Type your question..."
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            className="w-full border p-2 mb-2 rounded"
          />
          <button
            onClick={handleAskQuestion}
            className="bg-purple-500 text-white px-4 py-2 rounded"
          >
            Ask
          </button>
        </div>

        <div className="mt-2">
          {chatHistory.map((msg, idx) => (
            <div key={idx} className={`p-2 mb-1 rounded ${msg.role === "user" ? "bg-blue-100" : "bg-gray-200"}`}>
              <strong>{msg.role === "user" ? "You:" : "Bot:"}</strong> {msg.content}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

export default App;
