import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Modal, Form, Badge, Spinner, Alert } from 'react-bootstrap';
import { useNavigate } from 'react-router-dom';
import { coursesAPI } from '../services/api';

export default function Dashboard() {
  const [courses, setCourses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [showModal, setShowModal] = useState(false);
  const [newCourse, setNewCourse] = useState({ name: '', description: '' });
  const [creating, setCreating] = useState(false);
  const navigate = useNavigate();

  const fetchCourses = async () => {
    try {
      const res = await coursesAPI.list();
      setCourses(res.data.courses);
    } catch (err) {
      setError('Failed to load courses');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchCourses(); }, []);

  const handleCreate = async (e) => {
    e.preventDefault();
    setCreating(true);
    try {
      await coursesAPI.create(newCourse);
      setShowModal(false);
      setNewCourse({ name: '', description: '' });
      fetchCourses();
    } catch (err) {
      setError('Failed to create course');
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Delete this course and all its data?')) return;
    try {
      await coursesAPI.delete(id);
      fetchCourses();
    } catch (err) {
      setError('Failed to delete course');
    }
  };

  if (loading) return <div className="text-center mt-5"><Spinner animation="border" /></div>;

  return (
    <Container className="py-4">
      <div className="d-flex justify-content-between align-items-center mb-4">
        <div>
          <h2>📚 My Courses</h2>
          <p className="text-muted">Manage your courses and study materials</p>
        </div>
        <Button variant="primary" size="lg" onClick={() => setShowModal(true)}>
          + New Course
        </Button>
      </div>

      {error && <Alert variant="danger" onClose={() => setError('')} dismissible>{error}</Alert>}

      {courses.length === 0 ? (
        <Card className="text-center p-5">
          <Card.Body>
            <h4>No courses yet</h4>
            <p className="text-muted">Create your first course to start uploading materials and studying.</p>
            <Button variant="primary" onClick={() => setShowModal(true)}>Create Course</Button>
          </Card.Body>
        </Card>
      ) : (
        <Row xs={1} md={2} lg={3} className="g-4">
          {courses.map(course => (
            <Col key={course.id}>
              <Card className="h-100 shadow-sm" style={{ cursor: 'pointer' }}>
                <Card.Body onClick={() => navigate(`/courses/${course.id}`)}>
                  <Card.Title>{course.name}</Card.Title>
                  <Card.Text className="text-muted">
                    {course.description || 'No description'}
                  </Card.Text>
                  <Badge bg="info">{course.document_count} documents</Badge>
                </Card.Body>
                <Card.Footer className="d-flex justify-content-between">
                  <small className="text-muted">
                    Created {new Date(course.created_at).toLocaleDateString()}
                  </small>
                  <Button variant="outline-danger" size="sm" onClick={(e) => { e.stopPropagation(); handleDelete(course.id); }}>
                    Delete
                  </Button>
                </Card.Footer>
              </Card>
            </Col>
          ))}
        </Row>
      )}

      {/* Create Course Modal */}
      <Modal show={showModal} onHide={() => setShowModal(false)}>
        <Modal.Header closeButton>
          <Modal.Title>Create New Course</Modal.Title>
        </Modal.Header>
        <Form onSubmit={handleCreate}>
          <Modal.Body>
            <Form.Group className="mb-3">
              <Form.Label>Course Name</Form.Label>
              <Form.Control
                type="text"
                placeholder="e.g., Data Structures & Algorithms"
                value={newCourse.name}
                onChange={e => setNewCourse({ ...newCourse, name: e.target.value })}
                required
              />
            </Form.Group>
            <Form.Group>
              <Form.Label>Description (optional)</Form.Label>
              <Form.Control
                as="textarea"
                rows={3}
                placeholder="Course description..."
                value={newCourse.description}
                onChange={e => setNewCourse({ ...newCourse, description: e.target.value })}
              />
            </Form.Group>
          </Modal.Body>
          <Modal.Footer>
            <Button variant="secondary" onClick={() => setShowModal(false)}>Cancel</Button>
            <Button variant="primary" type="submit" disabled={creating}>
              {creating ? <Spinner size="sm" /> : 'Create Course'}
            </Button>
          </Modal.Footer>
        </Form>
      </Modal>
    </Container>
  );
}
