import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Tab, Tabs, Badge, Spinner, Alert, Button } from 'react-bootstrap';
import { useParams, useNavigate } from 'react-router-dom';
import { coursesAPI, documentsAPI } from '../services/api';
import DocumentUpload from '../components/DocumentUpload';
import ChatInterface from '../components/ChatInterface';
import StudyPlanView from './StudyPlan';

export default function CourseDetail() {
  const { courseId } = useParams();
  const navigate = useNavigate();
  const [course, setCourse] = useState(null);
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState('chat');

  const fetchCourse = async () => {
    try {
      const res = await coursesAPI.get(courseId);
      setCourse(res.data);
    } catch (err) {
      navigate('/dashboard');
    }
  };

  const fetchDocuments = async () => {
    try {
      const res = await documentsAPI.list(courseId);
      setDocuments(res.data.documents);
    } catch (err) {
      console.error('Failed to load documents');
    }
  };

  useEffect(() => {
    Promise.all([fetchCourse(), fetchDocuments()]).then(() => setLoading(false));
  }, [courseId]);

  if (loading) return <div className="text-center mt-5"><Spinner animation="border" /></div>;
  if (!course) return <Alert variant="danger">Course not found</Alert>;

  const statusColor = { uploaded: 'secondary', processing: 'warning', ready: 'success', failed: 'danger' };

  return (
    <Container className="py-4">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <div>
          <Button variant="link" className="p-0 mb-2" onClick={() => navigate('/dashboard')}>
            ← Back to Courses
          </Button>
          <h2>{course.name}</h2>
          <p className="text-muted">{course.description || 'No description'}</p>
        </div>
        <Badge bg="info" className="fs-6">{documents.length} documents</Badge>
      </div>

      <Tabs activeKey={activeTab} onSelect={setActiveTab} className="mb-4">
        <Tab eventKey="chat" title="💬 Chat">
          <ChatInterface courseId={courseId} />
        </Tab>

        <Tab eventKey="documents" title={`📄 Documents (${documents.length})`}>
          <Row className="mb-4">
            <Col>
              <DocumentUpload courseId={courseId} onUploadComplete={() => { fetchDocuments(); fetchCourse(); }} />
            </Col>
          </Row>
          {documents.length === 0 ? (
            <Alert variant="info">No documents uploaded yet. Upload your syllabus, notes, or PYQs above.</Alert>
          ) : (
            <Row xs={1} md={2} className="g-3">
              {documents.map(doc => (
                <Col key={doc.id}>
                  <Card>
                    <Card.Body>
                      <div className="d-flex justify-content-between">
                        <div>
                          <Card.Title className="fs-6">{doc.filename}</Card.Title>
                          <Badge bg="primary" className="me-2">{doc.doc_type}</Badge>
                          <Badge bg={statusColor[doc.status] || 'secondary'}>{doc.status}</Badge>
                        </div>
                      </div>
                      <small className="text-muted">
                        Uploaded {new Date(doc.created_at).toLocaleDateString()}
                      </small>
                    </Card.Body>
                  </Card>
                </Col>
              ))}
            </Row>
          )}
        </Tab>

        <Tab eventKey="studyplan" title="📋 Study Plan">
          <StudyPlanView courseId={courseId} />
        </Tab>
      </Tabs>
    </Container>
  );
}
