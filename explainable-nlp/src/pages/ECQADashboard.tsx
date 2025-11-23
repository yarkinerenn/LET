import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Row, Col, Card, Table, Alert, Spinner, Button, Badge } from 'react-bootstrap';
import axios from 'axios';
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip
} from 'recharts';
import LLMSelector from '../components/LLMSelector';

const COLORS = ['#0088FE', '#FF8042'];

interface ECQAResult {
  label: string;
  score: number;
  actualLabel?: string;
  original_data?: any;
  question?: string;
  choices?: string[];
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
  classification_type?: string; // 'classification_only', 'bert_only', or undefined (classify_and_explain)
  explanation_models?: Array<{ provider: string; model: string }>;
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
  const [expandedRow, setExpandedRow] = useState<number | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [detailRes, statsRes] = await Promise.all([
          axios.get(`http://localhost:5000/api/classification/${classificationId}`, { withCredentials: true }),
          axios.get(`http://localhost:5000/api/classification/stats/${classificationId}`, { withCredentials: true })
        ]);
        console.log(statsRes.data,'statsRes.data');
        console.log(detailRes.data,'detailRes.data');
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

  // Check if this classification was created via "classify and explain"
  // If classification_type is undefined, it means it was created via classify_and_explain
  // In that case, we should disable adding more LLMs since explanations were already generated
  const isClassifyAndExplain = !classification?.classification_type;

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
    // Refresh classification data to update current models display
    const detailRes = await axios.get(`http://localhost:5000/api/classification/${classificationId}`, { withCredentials: true });
    setClassification(detailRes.data);
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
          {/* Header */}
          <Row className="mb-4 align-items-center justify-content-between">
            <Col md="auto">
              <h2 className="mb-2">ECQA Classification Report</h2>
              <div className="d-flex gap-2 flex-wrap mb-4">
                <Badge bg="info">Method: {classification?.method?.toUpperCase()}</Badge>
                {classification?.provider && <Badge bg="secondary">Provider: {classification.provider}</Badge>}
                {classification?.model && <Badge bg="dark">Model: {classification.model}</Badge>}
                <Badge bg="warning" text="dark">Type: ECQA</Badge>
              </div>
              <Button
                variant="outline-secondary"
                onClick={() => navigate(`/dataset/${datasetId}`)}
              >
                ← Back to datasetview
              </Button>
            </Col>
            <Col md="auto">
              <LLMSelector 
                onModelsSubmit={handleModelsSubmit} 
                disabled={isClassifyAndExplain}
                buttonText={isClassifyAndExplain ? "Explanations Already Generated" : "Choose Different LLMs"}
                currentModels={classification?.explanation_models || []}
              />
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
                        // Normalize labels aggressively: trim, case fold, unicode normalize, collapse spaces, strip digits/punctuation
                        const normalize = (str?: string) =>
                          (str ?? '')
                            .normalize('NFKC')                 // Unicode normalize (handles NBSP-like characters)
                            .replace(/\u00A0/g, ' ')           // convert non‑breaking space to normal space
                            .replace(/[\u2000-\u200D\u2060]/g, '') // remove zero‑width spaces
                            .trim()
                            .toLowerCase()
                            .replace(/[.\d]/g, '')
                            .replace(/\s+/g, ' ');

                        // Map various actualLabel forms to the textual choice:
                        // - "1".."5"  -> q_op1..q_op5
                        // - "A".."E"  -> q_op1..q_op5
                        // - "1: text" -> "text"
                        // - "Answer: text" -> "text"
                        const resolveActualLabel = (label: string | undefined, choices: (string | undefined)[]) => {
                          if (label == null) return undefined;
                          const s = String(label).normalize('NFKC').trim();

                          // 1) numeric index
                          if (/^\d+$/.test(s)) {
                            const idx = parseInt(s, 10) - 1;
                            return choices[idx] ?? s;
                          }

                          // 2) letter index A-E
                          if (/^[A-E]$/i.test(s)) {
                            const idx = s.toUpperCase().charCodeAt(0) - 65; // A->0
                            return choices[idx] ?? s;
                          }

                          // 3) "1: something" or "3) something"
                          const m = s.match(/^(\d)\s*[:.)-]?\s*(.*)$/);
                          if (m) {
                            const idx = parseInt(m[1], 10) - 1;
                            if (m[2]) return m[2];
                            return choices[idx] ?? s;
                          }

                          // 4) "Answer: ..." prefix
                          return s.replace(/^answer\s*[:\-]\s*/i, '');
                        };

                        // ECQA data structure: question and choices are stored directly, not in original_data
                        const question = result.question || result.original_data?.q_text || '';
                        const choices = result.choices || [
                          result.original_data?.q_op1,
                          result.original_data?.q_op2,
                          result.original_data?.q_op3,
                          result.original_data?.q_op4,
                          result.original_data?.q_op5
                        ].filter(Boolean);

                        const normalizedPred = normalize(result.label);
                        const resolvedActual = resolveActualLabel(result.actualLabel, choices);
                        const normalizedActual = resolvedActual !== undefined ? normalize(resolvedActual) : undefined;
                        const isMismatch = normalizedActual !== undefined && normalizedPred !== normalizedActual;

                        // Find which choice index matches the predicted/actual labels
                        const findChoiceIndex = (label: string, choices: string[]) => {
                          const normalizedLabel = normalize(label);
                          return choices.findIndex(choice => normalize(choice) === normalizedLabel);
                        };

                        const predChoiceIdx = findChoiceIndex(result.label, choices);
                        const actualChoiceIdx = resolvedActual ? findChoiceIndex(resolvedActual, choices) : -1;

                        return (
                          <tr
                            key={index}
                            onClick={() => navigate(`/datasets/${datasetId}/classifications_ecqa/${classificationId}/results/${index}`)}
                            style={{ cursor: 'pointer' }}
                          >
                            <td style={{ padding: '1rem', verticalAlign: 'top', maxWidth: '400px' }}>
                              <div className="fw-semibold mb-2" style={{ color: '#495057', fontSize: '0.9rem' }}>
                                Question:
                              </div>
                              <div style={{ 
                                fontSize: '0.95rem', 
                                lineHeight: '1.6',
                                color: '#212529',
                                whiteSpace: 'normal',
                                wordBreak: 'break-word'
                              }}>
                                {(expandedRow === index
                                  ? question
                                  : (question || '').slice(0, 200) + ((question || '').length > 200 ? '...' : '')
                                )}
                                {(question || '').length > 200 && (
                                  <Button
                                    variant="link"
                                    size="sm"
                                    onClick={e => { e.stopPropagation(); setExpandedRow(expandedRow === index ? null : index); }}
                                    style={{ padding: '0 4px', marginLeft: 4, fontSize: '0.85rem' }}
                                  >
                                    {expandedRow === index ? "Show Less" : "Show More"}
                                  </Button>
                                )}
                              </div>
                            </td>
                            <td>
                              <Badge bg="info">{result.label?.toLowerCase()}</Badge>
                            </td>
                            <td style={{ padding: '1rem', verticalAlign: 'top' }}>
                              <div className="mb-2 fw-semibold" style={{ color: '#495057', fontSize: '0.9rem' }}>
                                Choices:
                              </div>
                              <div style={{ fontSize: '0.9rem' }}>
                                {choices.map((choice, i) => {
                                  const isPredicted = i === predChoiceIdx;
                                  const isActual = i === actualChoiceIdx && actualChoiceIdx >= 0;
                                  return choice ? (
                                    <div
                                      key={i}
                                      className="mb-2 p-2 rounded"
                                      style={{
                                        backgroundColor: isActual && isPredicted ? '#d1e7dd' : isActual ? '#f8d7da' : isPredicted ? '#cfe2ff' : '#f8f9fa',
                                        border: isPredicted || isActual ? '2px solid' : '1px solid #dee2e6',
                                        borderColor: isActual && isPredicted ? '#198754' : isActual ? '#dc3545' : isPredicted ? '#0d6efd' : '#dee2e6',
                                        transition: 'all 0.2s',
                                        lineHeight: '1.5'
                                      }}
                                    >
                                      <span className="fw-bold me-2" style={{ color: '#495057' }}>
                                        {String.fromCharCode(65 + i)}:
                                      </span>
                                      <span style={{ color: '#212529' }}>{choice}</span>
                                      {isPredicted && (
                                        <Badge bg="info" className="ms-2" style={{ fontSize: '0.75rem' }}>Predicted</Badge>
                                      )}
                                      {isActual && (
                                        <Badge bg="success" className="ms-2" style={{ fontSize: '0.75rem' }}>Actual</Badge>
                                      )}
                                    </div>
                                  ) : null;
                                })}
                              </div>
                            </td>
                            {result.actualLabel !== undefined && (
                              <td>
                                <Badge bg="primary">{result.actualLabel?.toLowerCase()}</Badge>
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