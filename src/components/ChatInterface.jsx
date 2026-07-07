import React, { useState, useEffect, useRef } from 'react';
import { Card, Form, Button, Spinner, Badge, Alert } from 'react-bootstrap';
import { queryAPI } from '../services/api';

export default function ChatInterface({ courseId }) {
  const [messages, setMessages] = useState([]);
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(true);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await queryAPI.history(courseId);
        const history = res.data.messages.reverse().flatMap(m => [
          { role: 'user', content: m.query },
          { role: 'assistant', content: m.response, citations: m.citations_json },
        ]);
        setMessages(history);
      } catch (err) {
        // No history yet
      } finally {
        setLoadingHistory(false);
      }
    };
    fetchHistory();
  }, [courseId]);

  useEffect(() => { scrollToBottom(); }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userMsg = { role: 'user', content: query };
    setMessages(prev => [...prev, userMsg]);
    setQuery('');
    setLoading(true);

    try {
      const res = await queryAPI.ask(courseId, query);
      const assistantMsg = {
        role: 'assistant',
        content: res.data.response,
        citations: res.data.citations,
        evaluation: res.data.evaluation,
        latency: res.data.latency_ms,
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, something went wrong. Please try again.',
        error: true,
      }]);
    } finally {
      setLoading(false);
    }
  };

  if (loadingHistory) return <div className="text-center"><Spinner animation="border" /></div>;

  return (
    <div>
      {/* Messages */}
      <div style={{ height: '500px', overflowY: 'auto', border: '1px solid #dee2e6', borderRadius: '8px', padding: '16px', marginBottom: '16px', backgroundColor: '#f8f9fa' }}>
        {messages.length === 0 && (
          <div className="text-center text-muted mt-5">
            <h5>👋 Ask a question about your course!</h5>
            <p>Upload materials first, then ask questions here.</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`mb-3 d-flex ${msg.role === 'user' ? 'justify-content-end' : 'justify-content-start'}`}>
            <Card style={{ maxWidth: '80%' }} className={msg.role === 'user' ? 'bg-primary text-white' : msg.error ? 'bg-danger text-white' : 'bg-white'}>
              <Card.Body className="py-2 px-3">
                <div style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</div>
                {msg.citations?.length > 0 && (
                  <div className="mt-2 pt-2 border-top">
                    <small><strong>Sources:</strong></small>
                    {msg.citations.map((c, j) => (
                      <Badge key={j} bg="light" text="dark" className="ms-1">
                        📄 {c.source} {c.year ? `(${c.year})` : ''}
                      </Badge>
                    ))}
                  </div>
                )}
                {msg.evaluation?.score && (
                  <div className="mt-1">
                    <small className="text-muted">
                      Quality: {msg.evaluation.score}/10 • {Math.round(msg.latency)}ms
                    </small>
                  </div>
                )}
              </Card.Body>
            </Card>
          </div>
        ))}
        {loading && (
          <div className="d-flex justify-content-start mb-3">
            <Card style={{ maxWidth: '80%' }}>
              <Card.Body className="py-2 px-3">
                <Spinner size="sm" className="me-2" />
                Analyzing your question through 7 AI agents...
              </Card.Body>
            </Card>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <Form onSubmit={handleSubmit} className="d-flex gap-2">
        <Form.Control
          type="text"
          placeholder="Ask a question about your course material..."
          value={query}
          onChange={e => setQuery(e.target.value)}
          disabled={loading}
          size="lg"
        />
        <Button type="submit" variant="primary" size="lg" disabled={loading || !query.trim()}>
          {loading ? <Spinner size="sm" /> : 'Send'}
        </Button>
      </Form>
    </div>
  );
}
