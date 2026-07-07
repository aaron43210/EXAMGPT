import React, { useState, useEffect } from 'react';
import { Card, Button, Form, Spinner, Alert, Badge } from 'react-bootstrap';
import { studyPlanAPI } from '../services/api';

export default function StudyPlanView({ courseId }) {
  const [plan, setPlan] = useState(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState('');
  const [days, setDays] = useState(14);
  const [hoursPerDay, setHoursPerDay] = useState(4);

  const fetchPlan = async () => {
    setLoading(true);
    try {
      const res = await studyPlanAPI.get(courseId);
      setPlan(res.data.plan_data);
    } catch (err) {
      // No plan yet — that's okay
      setPlan(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchPlan(); }, [courseId]);

  const handleGenerate = async (e) => {
    e.preventDefault();
    setGenerating(true);
    setError('');
    try {
      const res = await studyPlanAPI.generate(courseId, { days, hours_per_day: hoursPerDay });
      setPlan(res.data.plan_data);
    } catch (err) {
      setError('Failed to generate study plan. Make sure you have uploaded documents first.');
    } finally {
      setGenerating(false);
    }
  };

  if (loading) return <div className="text-center"><Spinner animation="border" /></div>;

  return (
    <div>
      {/* Generation Form */}
      <Card className="mb-4">
        <Card.Body>
          <Card.Title>Generate Study Plan</Card.Title>
          <Form onSubmit={handleGenerate} className="d-flex gap-3 align-items-end">
            <Form.Group>
              <Form.Label>Days until exam</Form.Label>
              <Form.Control type="number" value={days} onChange={e => setDays(Number(e.target.value))} min={1} max={90} />
            </Form.Group>
            <Form.Group>
              <Form.Label>Hours/day</Form.Label>
              <Form.Control type="number" value={hoursPerDay} onChange={e => setHoursPerDay(Number(e.target.value))} min={1} max={16} step={0.5} />
            </Form.Group>
            <Button type="submit" variant="primary" disabled={generating}>
              {generating ? <><Spinner size="sm" /> Generating...</> : '🎯 Generate Plan'}
            </Button>
          </Form>
        </Card.Body>
      </Card>

      {error && <Alert variant="danger">{error}</Alert>}

      {/* Display Plan */}
      {plan && (
        <div>
          <h4>{plan.title || 'Your Study Plan'}</h4>
          <p className="text-muted">
            {plan.total_days} days • {plan.hours_per_day} hours/day • {plan.total_days * plan.hours_per_day} total hours
          </p>

          {plan.priority_topics?.length > 0 && (
            <Card className="mb-3">
              <Card.Body>
                <Card.Title>🔥 Priority Topics</Card.Title>
                <div className="d-flex flex-wrap gap-2">
                  {plan.priority_topics.map((t, i) => (
                    <Badge key={i} bg="danger" className="fs-6">{t}</Badge>
                  ))}
                </div>
              </Card.Body>
            </Card>
          )}

          {plan.phases?.map((phase, i) => (
            <Card key={i} className="mb-3">
              <Card.Header>
                <strong>Phase {phase.phase}: {phase.name}</strong>
                <Badge bg="secondary" className="ms-2">{phase.days}</Badge>
                <Badge bg="info" className="ms-2">{phase.hours}h</Badge>
              </Card.Header>
              <Card.Body>
                <p><strong>Focus:</strong> {phase.focus}</p>
                {phase.topics?.length > 0 && (
                  <div className="mb-2">
                    <strong>Topics:</strong>{' '}
                    {phase.topics.map((t, j) => <Badge key={j} bg="primary" className="me-1">{t}</Badge>)}
                  </div>
                )}
                {phase.activities?.length > 0 && (
                  <ul className="mb-0">
                    {phase.activities.map((a, j) => <li key={j}>{a}</li>)}
                  </ul>
                )}
              </Card.Body>
            </Card>
          ))}

          {plan.tips?.length > 0 && (
            <Card className="mb-3">
              <Card.Body>
                <Card.Title>💡 Tips</Card.Title>
                <ul className="mb-0">
                  {plan.tips.map((t, i) => <li key={i}>{t}</li>)}
                </ul>
              </Card.Body>
            </Card>
          )}
        </div>
      )}

      {!plan && !loading && (
        <Alert variant="info">No study plan yet. Upload your materials and generate one above!</Alert>
      )}
    </div>
  );
}
