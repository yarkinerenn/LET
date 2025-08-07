import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Row, Col, Card, Table, Alert, Spinner, Button, Badge, Modal, Form } from 'react-bootstrap';
import axios from 'axios';
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip
} from 'recharts';

const COLORS = ['#0088FE', '#FF8042'];

interface ECQAResult {
  label: string;
  score: number;
  actualLabel?: string;
  original_data?: any;
}

interface ECQAStats {
  total: number;
  correct?: number;
  incorrect?: number;
  accuracy?: number;
}

interface ECQAData {
  _id: string;
  dataset_id: string;
  user_id: string;
  method: string;
  provider?: string;
  model?: string;
  results: ECQAResult[];
  created_at: string;
  stats: ECQAStats;
  data_type?: string; // 'ecqa'
}

const ECQADashboard = () => {
  const { datasetId, classificationId } = useParams<{ datasetId: string, classificationId: string }>();
  const [classification, setClassification] = useState<ECQAData | null>(null);
  const [stats, setStats] = useState<ECQAStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;
  const navigate = useNavigate();

  // Modal to show all choices if needed
  const [expandedRow, setExpandedRow] = useState<number | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [detailRes, statsRes] = await Promise.all([
          axios.get(`http://localhost:5000/api/classification/${classificationId}`, { withCredentials: true }),
          axios.get(`http://localhost:5000/api/classification/stats/${classificationId}`, { withCredentials: true })
        ]);
        setClassification(detailRes.data);
        setStats(statsRes.data.stats);
        setLoading(false);
      } catch (err) {
        setError("Failed to load classification data");
        setLoading(false);
      }
    };
    fetchData();
  }, [classificationId]);

  // Pagination
  const paginatedResults = classification?.results?.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  // Pie data
  const pieData = [
    { name: "Correct", value: stats?.correct || 0 },
    { name: "Incorrect", value: stats?.incorrect || 0 }
  ];

  return (
    <Container fluid className="py-4">
      {loading ? (
        <div className="text-center"><Spinner animation="border" /></div>
      ) : error ? (
        <Alert variant="danger">{error}</Alert>
      ) : (
        <>
          {/* Header */}
          <Row className="mb-4 align-items-center justify-content-between">
            <Col md="auto">
              <h2 className="mb-2">ECQA Classification Report</h2>
              <div className="d-flex gap-2 flex-wrap">
                <Badge bg="info">Method: {classification?.method?.toUpperCase()}</Badge>
                {classification?.provider && <Badge bg="secondary">Provider: {classification.provider}</Badge>}
                {classification?.model && <Badge bg="dark">Model: {classification.model}</Badge>}
                <Badge bg="warning" text="dark">Type: ECQA</Badge>
              </div>
            </Col>
          </Row>

          {/* Stats */}
          <Row className="mb-4">
            <Col md={3}>
              <Card className="mb-3">
                <Card.Body>
                  <Card.Title>Total Samples</Card.Title>
                  <Card.Text className="display-6">{stats?.total}</Card.Text>
                </Card.Body>
              </Card>
            </Col>
            <Col md={3}>
              <Card className="mb-3">
                <Card.Body>
                  <Card.Title>Correct</Card.Title>
                  <Card.Text className="display-6 text-success">{stats?.correct}</Card.Text>
                </Card.Body>
              </Card>
            </Col>
            <Col md={3}>
              <Card className="mb-3">
                <Card.Body>
                  <Card.Title>Incorrect</Card.Title>
                  <Card.Text className="display-6 text-danger">{stats?.incorrect}</Card.Text>
                </Card.Body>
              </Card>
            </Col>
            <Col md={3}>
              <Card className="mb-3">
                <Card.Body>
                  <Card.Title>Accuracy</Card.Title>
                  <Card.Text className="display-6 text-primary">
                    {(stats?.accuracy ? stats.accuracy * 100 : 0).toFixed(1)}%
                  </Card.Text>
                </Card.Body>
              </Card>
            </Col>
          </Row>

          {/* Pie Chart */}
          <Row className="mb-4">
            <Col md={6}>
              <Card className="h-100">
                <Card.Body>
                  <Card.Title>Correct vs Incorrect</Card.Title>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={80}
                        paddingAngle={5}
                        dataKey="value"
                      >
                        {pieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </Card.Body>
              </Card>
            </Col>
          </Row>

          {/* Predictions Table */}
          <Row>
            <Col>
              <Card>
                <Card.Body>
                  <Card.Title>Predictions</Card.Title>
                  <Table striped hover responsive>
                    <thead>
                      <tr>
                        <th>Question</th>
                        <th>Predicted Answer</th>
                        <th>All Choices</th>
                        {classification?.results[0]?.actualLabel !== undefined && <th>Actual Label</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedResults?.map((result, index) => {
                        const isMismatch =
                          result.actualLabel !== undefined &&
                          result.label !== result.actualLabel;
                        const original = result.original_data || {};
                        const choices = [
                          original.q_op1, original.q_op2, original.q_op3, original.q_op4, original.q_op5
                        ];
                        return (
                          <tr
                            key={index}
                            onClick={() => navigate(`/datasets/${datasetId}/classifications_ecqa/${classificationId}/results/${index}`)}
                            className={isMismatch ? 'table-danger' : ''}
                            style={{ cursor: 'pointer' }}
                          >
                            <td style={{ maxWidth: '400px', whiteSpace: 'normal' }}>
                              {(expandedRow === index
                                ? original.q_text
                                : (original.q_text || '').slice(0, 180) + ((original.q_text || '').length > 180 ? '...' : '')
                              )}
                              {(original.q_text || '').length > 180 && (
                                <Button
                                  variant="link"
                                  size="sm"
                                  onClick={e => { e.stopPropagation(); setExpandedRow(expandedRow === index ? null : index); }}
                                  style={{ padding: 0, marginLeft: 4 }}
                                >
                                  {expandedRow === index ? "Show Less" : "Show More"}
                                </Button>
                              )}
                            </td>
                            <td>
                              <Badge bg="info">{result.label}</Badge>
                            </td>
                            <td>
                              <ul className="mb-0">
                                {choices.map((choice, i) =>
                                  choice && <li key={i}><strong>{i + 1}:</strong> {choice}</li>
                                )}
                              </ul>
                            </td>
                            {result.actualLabel !== undefined && (
                              <td>
                                <Badge bg="primary">{result.actualLabel}</Badge>
                              </td>
                            )}
                          </tr>
                        );
                      })}
                    </tbody>
                  </Table>
                  {/* Pagination controls */}
                  <div className="d-flex justify-content-center">
                    <Button
                      variant="outline-primary"
                      disabled={currentPage === 1}
                      onClick={() => setCurrentPage(p => p - 1)}
                    >Previous</Button>
                    <span className="mx-3 my-auto">
                      Page {currentPage} of {Math.ceil((classification?.results?.length || 0) / itemsPerPage)}
                    </span>
                    <Button
                      variant="outline-primary"
                      disabled={currentPage * itemsPerPage >= (classification?.results?.length || 0)}
                      onClick={() => setCurrentPage(p => p + 1)}
                    >Next</Button>
                  </div>
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </>
      )}
    </Container>
  );
};

export default ECQADashboard;