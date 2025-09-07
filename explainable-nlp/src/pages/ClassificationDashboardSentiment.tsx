import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Row, Col, Card, Table, Alert, Spinner, Button, Badge } from 'react-bootstrap';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, PieChart, Pie, Cell, ResponsiveContainer
} from 'recharts';
import LLMSelector from '../components/LLMSelector';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];


interface ClassificationResult {
  text?: string;
  label: string | number;
  score: number;
  actualLabel?: string | number;
}

interface ClassificationStats {
  total: number;
  positive: number;
  negative: number;
  correct?: number;
  incorrect?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
}

interface ClassificationData {
  _id: string;
  dataset_id: string;
  user_id: string;
  method: string;
  provider?: string;
  model?: string;
  results: ClassificationResult[];
  created_at: string;
  stats: ClassificationStats;
}

function toSentiment(val: string | number | undefined): "POSITIVE" | "NEGATIVE" | undefined {
  if (val === 1 || val === "1" || val === "POSITIVE") return "POSITIVE";
  if (val === 0 || val === "0" || val === "NEGATIVE") return "NEGATIVE";
  return undefined;
}

const SentimentDashboard = () => {
  const { datasetId, classificationId } = useParams<{ datasetId: string, classificationId: string }>();
  const [classification, setClassification] = useState<ClassificationData | null>(null);
  const [stats, setStats] = useState<ClassificationStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [detailRes, statsRes] = await Promise.all([
          axios.get(`http://localhost:5000/api/classification/${classificationId}`, { withCredentials: true }),
          axios.get(`http://localhost:5000/api/classification/stats/${classificationId}`, { withCredentials: true })
        ]);
        setClassification(detailRes.data);
        console.log(detailRes.data);
        setStats(statsRes.data.stats);
        setLoading(false);
      } catch (err) {
        setError("Failed to load classification data");
        setLoading(false);
      }
    };
    fetchData();
  }, [classificationId]);

  const paginatedResults = classification?.results?.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const pieData = [
    { name: "Positive", value: stats?.positive || 0 },
    { name: "Negative", value: stats?.negative || 0 }
  ];

  const handleModelsSubmit = async (selectedModels: string[]) => {
    const explanation_models = selectedModels.map(model => {
      const [provider, ...rest] = model.split(':');
      return { provider, model: rest.join(':') };
    });
    await axios.post(
      `http://localhost:5000/api/classification/${classificationId}/add_explanation_models`,
      { explanation_models },
      { withCredentials: true }
    );
    alert('Explanation models added successfully!');
  };

  return (
    <Container fluid className="py-4">
      {loading ? (
        <div className="text-center"><Spinner animation="border" /></div>
      ) : error ? (
        <Alert variant="danger">{error}</Alert>
      ) : (
        <>
          <Button
            variant="outline-secondary"
            onClick={() => navigate(`/dataset/${datasetId}`)}
          >
            ← Back to dataset view
          </Button>
          
          <Row className="mb-4 align-items-center justify-content-between">
            <Col md="auto">
              <h2 className="mb-2">Sentiment Analysis Report</h2>
              <div className="d-flex gap-2 flex-wrap">
                <Badge bg="info">Method: {classification?.method?.toUpperCase()}</Badge>
                {classification?.provider && <Badge bg="secondary">Provider: {classification.provider}</Badge>}
                {classification?.model && <Badge bg="dark">Model: {classification.model}</Badge>}
              </div>
            </Col>

            <Col md="auto">
              <LLMSelector onModelsSubmit={handleModelsSubmit} />
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
                  <Card.Title>Positive</Card.Title>
                  <Card.Text className="display-6 text-success">{stats?.positive}</Card.Text>
                </Card.Body>
              </Card>
            </Col>
            <Col md={3}>
              <Card className="mb-3">
                <Card.Body>
                  <Card.Title>Negative</Card.Title>
                  <Card.Text className="display-6 text-danger">{stats?.negative}</Card.Text>
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

          {/* Charts */}
          <Row className="mb-4">
            <Col md={6}>
              <Card className="h-100">
                <Card.Body>
                  <Card.Title>Sentiment Distribution</Card.Title>
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
            <Col md={6}>
              <Card>
                <Card.Body>
                  <Card.Title>Performance Metrics</Card.Title>
                  <ResponsiveContainer width="100%" height={250}>
                    <BarChart
                      data={[
                        { name: "F1 Score", value: stats?.f1_score || 0 },
                        { name: "Precision", value: stats?.precision || 0 },
                        { name: "Recall", value: stats?.recall || 0 },
                      ]}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis domain={[0, 1]} />
                      <Tooltip />
                      <Bar dataKey="value" fill="#8884d8" />
                    </BarChart>
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
                        <th>Text</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        {classification?.results[0]?.actualLabel !== undefined && <th>Actual Label</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedResults?.map((result, index) => {
                        const isMismatch =
                          result.actualLabel !== undefined &&
                          toSentiment(result.label) !== toSentiment(result.actualLabel);
                        return (
                          <tr
                            key={index}
                            onClick={() => navigate(`/datasets/${datasetId}/classifications/${classificationId}/results/${index}`)}
                            className={isMismatch ? 'table-danger' : ''}
                            style={{ cursor: 'pointer' }}
                          >
                            <td className="text-truncate" style={{ maxWidth: '300px' }}>{result.text}</td>
                            <td>
                              <Badge bg={toSentiment(result.label) === 'POSITIVE' ? 'success' : 'danger'}>
                                {toSentiment(result.label)}
                              </Badge>
                            </td>
                            <td>{(result.score * 100).toFixed(1)}%</td>
                            {result.actualLabel !== undefined && (
                              <td>
                                <Badge
                                  bg={toSentiment(result.actualLabel) === 'POSITIVE' ? 'success' : 'danger'}
                                >
                                  {toSentiment(result.actualLabel)}
                                </Badge>
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

export default SentimentDashboard;