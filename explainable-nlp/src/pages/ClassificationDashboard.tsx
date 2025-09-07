

import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Container, Row, Col, Card, Table, Alert, Spinner, Button, Badge, Modal, Form, Accordion } from 'react-bootstrap';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, PieChart, Pie, Cell, ResponsiveContainer
} from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

// Example model lists (update as needed)
const groqModels = [
        { name: " llama3-70b" },
        { name: "deepseek-r1-distill-llama-70b" },
        { name: "deepseek-r1-distill-qwen-32b" },
        { name: "gemma2-9b-it" },
        { name: "llama-3.1-8b-instant" },
        { name: "llama-3.2-11b-vision-preview" },
        { name: "llama-3.2-1b-preview" },
        { name: "llama-3.2-3b-preview" },
        { name: "llama-3.2-90b-vision-preview" },
        { name: "llama-3.3-70b-specdec" },
        { name: "llama-3.3-70b-versatile" },
        { name: "llama-guard-3-8b" },
        { name: "llama3-70b-8192" },
        { name: "llama3-8b-8192" },
        { name: "mistral-saba-24b" },
        { name: "qwen-2.5-32b" },
        { name: "qwen-2.5-coder-32b" },
        { name: "qwen-qwq-32b" }
    ];
    const openrouterModels = [
        { name: "deepseek/deepseek-r1-0528-qwen3-8b:free" },
        { name: "sarvamai/sarvam-m:free" },
        { name: "mistralai/devstral-small-2505:free" },
        { name: "devstral-small" },
        { name: "gemma-3n-e4b-it" },
        { name: "llama-3.3-8b-instruct" },
        { name: "deephermes-3-mistral-24b-preview" },
        { name: "phi-4-reasoning-plus" },
        { name: "phi-4-reasoning" },
        { name: "internvl3-14b" },
        { name: "internvl3-2b" },
        { name: "deepseek-prover-v2" },
        { name: "qwen3-30b-a3b" },
        { name: "qwen3-8b" },
        { name: "qwen3-14b" },
        { name: "qwen3-32b" },
        { name: "qwen3-235b-a22b" },
        { name: "deepseek-r1t-chimera" },
        { name: "mai-ds-r1" },
        { name: "glm-z1-32b" },
        { name: "glm-4-32b" },
        { name: "shisa-v2-llama3.3-70b" },
        { name: "qwq-32b-arliai-rpr-v1" },
        { name: "deepcoder-14b-preview" },
        { name: "kimi-vl-a3b-thinking" },
        { name: "llama-3.3-nemotron-super-49b-v1" },
        { name: "llama-3.1-nemotron-ultra-253b-v1" },
        { name: "llama-4-maverick" },
        { name: "llama-4-scout" },
        { name: "deepseek-v3-base" },
        { name: "qwen2.5-vl-3b-instruct" },
        { name: "gemini-2.5-pro-exp-03-25" },
        { name: "qwen2.5-vl-32b-instruct" },
        { name: "deepseek-chat-v3-0324" },
        { name: "qwerky-72b" },
        { name: "mistral-small-3.1-24b-instruct" },
        { name: "olympiccoder-32b" },
        { name: "gemma-3-1b-it" },
        { name: "gemma-3-4b-it" },
        { name: "gemma-3-12b-it" },
        { name: "reka-flash-3" },
        { name: "gemma-3-27b-it" },
        { name: "deepseek-r1-zero" },
        { name: "qwq-32b" },
        { name: "moonlight-16b-a3b-instruct" },
        { name: "deephermes-3-llama-3-8b-preview" },
        { name: "dolphin3.0-r1-mistral-24b" },
        { name: "dolphin3.0-mistral-24b" },
        { name: "qwen2.5-vl-72b-instruct" },
        { name: "mistral-small-24b-instruct-2501" },
        { name: "deepseek-r1-distill-qwen-32b" },
        { name: "deepseek-r1-distill-qwen-14b" },
        { name: "deepseek-r1-distill-llama-70b" },
        { name: "deepseek-r1" },
        { name: "deepseek-chat" },
        { name: "gemini-2.0-flash-exp" },
        { name: "llama-3.3-70b-instruct" },
        { name: "qwen-2.5-coder-32b-instruct" },
        { name: "qwen-2.5-7b-instruct" },
        { name: "llama-3.2-3b-instruct" },
        { name: "llama-3.2-1b-instruct" },
        { name: "llama-3.2-11b-vision-instruct" },
        { name: "qwen-2.5-72b-instruct" },
        { name: "qwen-2.5-vl-7b-instruct" },
        { name: "llama-3.1-405b" },
        { name: "llama-3.1-8b-instruct" },
        { name: "mistral-nemo" },
        { name: "gemma-2-9b-it" },
        { name: "mistral-7b-instruct" }
    ];

interface ClassificationResult {
  text?: string;
  label: string | number;
  score: number;
  actualLabel?: string | number;
  original_data?: any;
  citing_prompt?: string;
  holdings?: string[];
}

interface ClassificationStats {
  total: number;
  positive?: number;
  negative?: number;
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
  data_type?: string; // 'sentiment' or 'legal'
}
function toSentiment(val: string | number | undefined): "POSITIVE" | "NEGATIVE" | undefined {
  if (val === 1 || val === "POSITIVE") return "POSITIVE";
  if (val === 0 || val === "NEGATIVE") return "NEGATIVE";
  return undefined;
}

const ClassificationDashboard = () => {
  const { datasetId, classificationId } = useParams<{ datasetId: string, classificationId: string }>();
  const [classification, setClassification] = useState<ClassificationData | null>(null);
  const [stats, setStats] = useState<ClassificationStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;
  const navigate = useNavigate();

  // Model Modal
  const [showModelModal, setShowModelModal] = useState(false);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
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

  const dataType = classification?.data_type || "sentiment";

  // For charts and tables
  const paginatedResults = classification?.results?.slice(
    (currentPage - 1) * itemsPerPage,
    currentPage * itemsPerPage
  );

  const pieData =
    dataType === "sentiment"
      ? [
        { name: "Positive", value: stats?.positive || 0 },
        { name: "Negative", value: stats?.negative || 0 }
      ]
      : [
        { name: "Correct", value: stats?.correct || 0 },
        { name: "Incorrect", value: stats?.incorrect || 0 }
      ];

  // Add/Change LLM Models Modal handler (send to backend, update as needed)
  const handleSubmitModels = async () => {
    try {
      const explanation_models = selectedModels.map(model => {
        const [provider, ...rest] = model.split(':');
        return { provider, model: rest.join(':') };
      });
      await axios.post(
        `http://localhost:5000/api/classification/${classificationId}/add_explanation_models`,
        { explanation_models },
        { withCredentials: true }
      );
      setShowModelModal(false);
      alert('Explanation models added successfully!');
    } catch (error) {
      alert('Failed to add explanation models. Please try again.');
    }
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
                ← Back to datasetview
          </Button>
          <Row className="mb-4 align-items-center justify-content-between">
            <Col md="auto">
              <h2 className="mb-2">Classification Report</h2>
              <div className="d-flex gap-2 flex-wrap">
                <Badge bg="info">Method: {classification?.method?.toUpperCase()}</Badge>
                {classification?.provider && <Badge bg="secondary">Provider: {classification.provider}</Badge>}
                {classification?.model && <Badge bg="dark">Model: {classification.model}</Badge>}
                <Badge bg="warning" text="dark">Type: {dataType}</Badge>
              </div>
            </Col>

            <Col md="auto">
              <Button variant="outline-primary" onClick={() => { setSelectedModels([]); setShowModelModal(true); }}>
                Choose Different LLMs
              </Button>
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
            {dataType === 'sentiment' && (
              <>
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
              </>
            )}
            {dataType === 'legal' && (
              <>
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
              </>
            )}
          </Row>

          {/* Pie Chart and Bar Chart */}
          <Row className="mb-4">
            {dataType === "sentiment" && (
              <>
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
              </>
            )}
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
                        {dataType === 'sentiment' ? (
                          <>
                            <th>Text</th>
                            <th>Prediction</th>
                            <th>Confidence</th>
                            {classification?.results[0]?.actualLabel !== undefined && <th>Actual Label</th>}
                          </>
                        ) : (
                          <>
                            <th>Citing Prompt</th>
                            <th>Predicted Holding</th>
                            <th>All Holdings</th>
                            {classification?.results[0]?.actualLabel !== undefined && <th>Actual Label</th>}
                          </>
                        )}
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedResults?.map((result, index) => {
                        if (dataType === 'sentiment') {
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
                        } else {
                          // Legal/CaseHold display
                          const isMismatch = result.actualLabel !== undefined && String(result.label) !== String(result.actualLabel);
                          
                          // Get citing prompt from either direct field or original_data
                          const citingPrompt = result.citing_prompt || result.original_data?.citing_prompt || '';
                          
                          // Get holdings from either direct field or original_data
                          const holdings = result.holdings || [0,1,2,3,4].map(i => result.original_data?.[`holding_${i}`]).filter(Boolean);
                          
                          return (
                            <tr
                              key={index}
                              onClick={() => navigate(`/datasets/${datasetId}/classifications/${classificationId}/results/${index}`)}
                              className={isMismatch ? 'table-danger' : ''}
                              style={{ cursor: 'pointer' }}
                            >
                             <td style={{ maxWidth: '400px', whiteSpace: 'normal' }}>
                              {(expandedRow === index
                                ? citingPrompt
                                : citingPrompt.slice(0, 180) + (citingPrompt.length > 180 ? '...' : '')
                              )}
                              {citingPrompt.length > 180 && (
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
                                  {holdings.map((holding: string, i: number) => holding && (
                                    <li key={i}><strong>{i}:</strong> {holding}</li>
                                  ))}
                                </ul>
                              </td>
                              {result.actualLabel !== undefined && (
                                <td>
                                  <Badge bg="primary">{result.actualLabel}</Badge>
                                </td>
                              )}
                            </tr>
                          );
                        }
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

      {/* Model Modal */}
      <Modal show={showModelModal} onHide={() => setShowModelModal(false)} size="lg">
        <Modal.Header closeButton>
          <Modal.Title>Select Models for Explanation</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p className="text-muted">Select models from different providers. You can choose multiple models.</p>
          <Accordion defaultActiveKey="0">
            <Accordion.Item eventKey="0">
              <Accordion.Header>Groq ({selectedModels.filter(m => m.startsWith('groq:')).length} selected)</Accordion.Header>
              <Accordion.Body>
                <div className="row">
                  {groqModels.map((model, index) => {
                    const modelKey = `groq:${model.name}`;
                    return (
                      <div className="col-md-6 mb-2" key={modelKey}>
                        <Form.Check
                          type="checkbox"
                          id={`groq-${index}`}
                          label={model.name}
                          checked={selectedModels.includes(modelKey)}
                          onChange={(e) => {
                            const updated = e.target.checked
                              ? [...selectedModels, modelKey]
                              : selectedModels.filter((m) => m !== modelKey);
                            setSelectedModels(updated);
                          }}
                        />
                      </div>
                    );
                  })}
                </div>
              </Accordion.Body>
            </Accordion.Item>
            <Accordion.Item eventKey="1">
              <Accordion.Header>OpenRouter ({selectedModels.filter(m => m.startsWith('openrouter:')).length} selected)</Accordion.Header>
              <Accordion.Body>
                <div className="row">
                  {openrouterModels.map((model, index) => {
                    const modelKey = `openrouter:${model.name}`;
                    return (
                      <div className="col-md-6 mb-2" key={modelKey}>
                        <Form.Check
                          type="checkbox"
                          id={`openrouter-${index}`}
                          label={model.name}
                          checked={selectedModels.includes(modelKey)}
                          onChange={(e) => {
                            const updated = e.target.checked
                              ? [...selectedModels, modelKey]
                              : selectedModels.filter((m) => m !== modelKey);
                            setSelectedModels(updated);
                          }}
                        />
                      </div>
                    );
                  })}
                </div>
              </Accordion.Body>
            </Accordion.Item>
          </Accordion>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowModelModal(false)}>Cancel</Button>
          <Button variant="primary" onClick={handleSubmitModels} disabled={selectedModels.length === 0}>
            Submit
          </Button>
        </Modal.Footer>
      </Modal>
    </Container>
  );
};

export default ClassificationDashboard;