import React, { useState } from 'react';
import { Card, Form, Button, Spinner, Alert, ProgressBar } from 'react-bootstrap';
import { documentsAPI } from '../services/api';

export default function DocumentUpload({ courseId, onUploadComplete }) {
  const [file, setFile] = useState(null);
  const [docType, setDocType] = useState('notes');
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const handleUpload = async (e) => {
    e.preventDefault();
    if (!file) return;

    setUploading(true);
    setMessage('');
    setError('');

    try {
      const res = await documentsAPI.upload(courseId, file, docType);
      setMessage(`✅ ${res.data.filename} uploaded and indexed successfully!`);
      setFile(null);
      e.target.reset();
      if (onUploadComplete) onUploadComplete();
    } catch (err) {
      setError(err.response?.data?.detail || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <Card>
      <Card.Body>
        <Card.Title>📤 Upload Document</Card.Title>
        <Form onSubmit={handleUpload}>
          <div className="d-flex gap-3 align-items-end">
            <Form.Group className="flex-grow-1">
              <Form.Label>PDF File</Form.Label>
              <Form.Control
                type="file"
                accept=".pdf"
                onChange={e => setFile(e.target.files[0])}
                disabled={uploading}
              />
            </Form.Group>
            <Form.Group>
              <Form.Label>Type</Form.Label>
              <Form.Select value={docType} onChange={e => setDocType(e.target.value)} disabled={uploading}>
                <option value="notes">📝 Notes</option>
                <option value="syllabus">📋 Syllabus</option>
                <option value="pyq">📜 PYQ</option>
              </Form.Select>
            </Form.Group>
            <Button type="submit" variant="success" disabled={!file || uploading}>
              {uploading ? <><Spinner size="sm" /> Uploading...</> : 'Upload'}
            </Button>
          </div>
        </Form>
        {message && <Alert variant="success" className="mt-3 mb-0">{message}</Alert>}
        {error && <Alert variant="danger" className="mt-3 mb-0">{error}</Alert>}
      </Card.Body>
    </Card>
  );
}
