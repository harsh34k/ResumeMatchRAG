// src/components/ChatInterface.tsx
import React, { useState } from 'react';
import axios from 'axios';

interface ChatInterfaceProps {
    sessionId: string;
}

const ChatInterface: React.FC<ChatInterfaceProps> = ({ sessionId }) => {
    const [question, setQuestion] = useState('');
    const [messages, setMessages] = useState<any>([]);

    const handleSend = async () => {
        if (!question.trim()) return;

        let newMessages = [...messages, { text: question, sender: 'user' }];
        setMessages(newMessages);
        setQuestion('');

        try {
            const response = await axios.post(
                'http://localhost:8000/ask/',
                { question },
                { params: { session_id: sessionId } }
            );
            setMessages([
                ...newMessages,
                { text: response.data.response, sender: 'bot' },
            ]);
        } catch (error) {
            console.error(error);
            setMessages([
                ...newMessages,
                { text: 'Error fetching response from AI.', sender: 'bot' },
            ]);
        }
    };

    return (
        <div className="space-y-4">
            <div className="space-y-2">
                {messages.map((msg: any, index: any) => (
                    <div
                        key={index}
                        className={`p-3 rounded-lg ${msg.sender === 'user' ? 'bg-blue-50' : 'bg-gray-100'}`}
                    >
                        <p className="font-semibold">{msg.sender === 'user' ? 'You' : 'AI'}</p>
                        <p>{msg.text}</p>
                    </div>
                ))}
            </div>
            <div className="flex space-x-2">
                <input
                    type="text"
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    placeholder="Ask a question..."
                    className="flex-1 p-3 border border-gray-300 rounded-md"
                />
                <button
                    onClick={handleSend}
                    className="px-6 py-3 bg-gradient-to-r from-primary to-blue-600 text-white font-semibold rounded-lg shadow-md hover:from-blue-600 hover:to-primary transition-all duration-300"
                >
                    Send
                </button>
            </div>
        </div>
    );
};

export default ChatInterface;
