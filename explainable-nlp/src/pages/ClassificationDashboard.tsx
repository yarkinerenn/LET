import { useState, useEffect } from 'react';
import { useParams, Link,useNavigate } from 'react-router-dom';
import { Container, Row, Col, Card, Table, Alert, Spinner, Button, Badge,Modal,Form } from 'react-bootstrap';
import axios from 'axios';
import {
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    Legend as RechartsLegend,
    PieChart,
    Pie,
    Cell,
    ResponsiveContainer
} from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

interface ClassificationResult {
    text: string;
    label: string;
    score: number;
    actualLabel?: string;
    original_data?: any;
}

interface ClassificationStats {
    total: number;
    positive: number;
    negative: number;
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

const ClassificationDashboard = () => {
    const { datasetId, classificationId } = useParams<{ datasetId: string, classificationId: string }>();
    const [classification, setClassification] = useState<ClassificationData | null>(null);
    const [stats, setStats] = useState<ClassificationStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [currentPage, setCurrentPage] = useState(1);
    const itemsPerPage = 10;
    const navigate = useNavigate();
    const [showModelModal, setShowModelModal] = useState(false);
    const availableModels = [
      { provider: 'openrouter', model: "deepseek/deepseek-r1-0528-qwen3-8b:free" },
      { provider: 'grok', model: 'llama-3.1-8b-instant' },
      { provider: 'grok', model: 'mistral-saba-24b' },
      { provider: 'grok', model: 'qwen-2.5-32b' },
    ];
    const [selectedModels, setSelectedModels] = useState<string[]>([]);
    useEffect(() => {
        const fetchData = async () => {
            try {
                const [detailRes, statsRes] = await Promise.all([
                    axios.get(`http://localhost:5000/api/classification/${classificationId}`, {
                        withCredentials: true,
                    }),
                    axios.get(`http://localhost:5000/api/classification/stats/${classificationId}`, {
                        withCredentials: true,
                    })
                ]);
                const normalizedResults = detailRes.data.results.map((result: { actualLabel: number; }) => ({
                    ...result,
                    actualLabel: result.actualLabel === 1 ? 'POSITIVE' :
                        result.actualLabel === 0 ? 'NEGATIVE' :
                            result.actualLabel
                }));

                setClassification({
                    ...detailRes.data,
                    results: normalizedResults
                });                setStats(statsRes.data.stats);
                setLoading(false);
            } catch (err) {
                setError("Failed to load classification data");
                setLoading(false);
            }
        };

        fetchData();
    }, [classificationId]);

    const renderConfidence = (score: number) => {
        const percentage = (score * 100).toFixed(1);
        return (
            <div className="d-flex align-items-center">
                <div
                    style={{
                        width: '60px',
                        height: '10px',
                        backgroundColor: '#e0e0e0',
                        borderRadius: '5px',
                        marginRight: '8px'
                    }}
                >
                    <div
                        style={{
                            width: `${percentage}%`,
                            height: '100%',
                            backgroundColor: score > 0.7 ? '#4CAF50' : score > 0.4 ? '#FFC107' : '#F44336',
                            borderRadius: '5px'
                        }}
                    />
                </div>
                {percentage}%
            </div>
        );
    };

    const paginatedResults = classification?.results?.slice(
        (currentPage - 1) * itemsPerPage,
        currentPage * itemsPerPage
    );

const handleSubmitModels = async () => {
  try {
    const explanation_models = selectedModels.map(model => {
      const [provider, modelName] = model.split(':');
      return { provider, model: modelName };
    });

    await axios.post(
      `http://localhost:5000/api/classification/${classificationId}/add_explanation_models`,
      { explanation_models },
      { withCredentials: true }
    );

    setShowModelModal(false);
    alert('Explanation models added successfully!');
  } catch (error) {
    console.error('Failed to add explanation models:', error);
    alert('Failed to add explanation models. Please try again.');
  }
};

    return (

        <><Container fluid className="py-4">
            {loading ? (
                <div className="text-center">
                    <Spinner animation="border"/>
                </div>
            ) : error ? (
                <Alert variant="danger">{error}</Alert>
            ) : (
                <>
                    <Row className="mb-4 align-items-center justify-content-between">
              <Col md="auto">
                <h2 className="mb-2">Classification Report</h2>
                <div className="d-flex gap-2 flex-wrap">
                  <Badge bg="info">
                    Method: {classification?.method?.toUpperCase()}
                  </Badge>
                  {classification?.provider && (
                    <Badge bg="secondary">
                      Provider: {classification.provider}
                    </Badge>
                  )}
                  {classification?.model && (
                    <Badge bg="dark">
                      Model: {classification.model}
                    </Badge>
                  )}
                </div>
              </Col>

              <Col md="auto">
                <Button
                  variant="outline-primary"
                  onClick={() => {
                    setSelectedModels([]);
                    setShowModelModal(true);
                  }}
                  className="mt-2"
                >
                  Choose Different LLMs
                </Button>
              </Col>
            </Row>


                    <Row className="mb-4">
                        <Col md={3}>
                            <Card className="mb-3">
                                <Card.Body>
                                    <Card.Title>Total Samples</Card.Title>
                                    <Card.Text className="display-6">
                                        {stats?.total}
                                    </Card.Text>
                                </Card.Body>
                            </Card>
                        </Col>

                        <Col md={3}>
                            <Card className="mb-3">
                                <Card.Body>
                                    <Card.Title>Positive</Card.Title>
                                    <Card.Text className="display-6 text-success">
                                        {stats?.positive}
                                    </Card.Text>
                                </Card.Body>
                            </Card>
                        </Col>

                        <Col md={3}>
                            <Card className="mb-3">
                                <Card.Body>
                                    <Card.Title>Negative</Card.Title>
                                    <Card.Text className="display-6 text-danger">
                                        {stats?.negative}
                                    </Card.Text>
                                </Card.Body>
                            </Card>
                        </Col>

                        {stats?.accuracy && (
                            <Col md={3}>
                                <Card className="mb-3">
                                    <Card.Body>
                                        <Card.Title>Accuracy</Card.Title>
                                        <Card.Text className="display-6 text-primary">
                                            {(stats.accuracy * 100).toFixed(1)}%
                                        </Card.Text>
                                    </Card.Body>
                                </Card>
                            </Col>
                        )}
                    </Row>

                    <Row className="mb-4">
                        <Col md={6}>
                            <Card className="h-100">
                                <Card.Body>
                                    <Card.Title>Sentiment Distribution</Card.Title>
                                    <ResponsiveContainer width="100%" height={300}>
                                        <PieChart>
                                            <Pie
                                                data={[
                                                    {name: 'Positive', value: stats?.positive || 0},
                                                    {name: 'Negative', value: stats?.negative || 0},
                                                ]}
                                                cx="50%"
                                                cy="50%"
                                                innerRadius={60}
                                                outerRadius={80}
                                                paddingAngle={5}
                                                dataKey="value"
                                            >
                                                {['Positive', 'Negative'].map((entry, index) => (
                                                    <Cell
                                                        key={`cell-${index}`}
                                                        fill={index === 0 ? '#4CAF50' : '#F44336'}/>
                                                ))}
                                            </Pie>
                                            <Tooltip/>
                                        </PieChart>
                                    </ResponsiveContainer>
                                </Card.Body>
                            </Card>
                        </Col>

                        <Col md={6}>
                            <Card className="h-100">
                                <Card.Body>
                                    <Card.Title>Performance Metrics</Card.Title>
                                    <ResponsiveContainer width="100%" height={300}>
                                        <BarChart
                                            data={[
                                                {
                                                    name: 'Precision',
                                                    value: stats?.precision || 0,
                                                },
                                                {
                                                    name: 'Recall',
                                                    value: stats?.recall || 0,
                                                },
                                                {
                                                    name: 'F1 Score',
                                                    value: stats?.f1_score || 0,
                                                },
                                            ]}
                                        >
                                            <CartesianGrid strokeDasharray="3 3"/>
                                            <XAxis dataKey="name"/>
                                            <YAxis/>
                                            <Tooltip/>
                                            <RechartsLegend/>
                                            <Bar dataKey="value" fill="#8884d8"/>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </Card.Body>
                            </Card>
                        </Col>
                    </Row>

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
                                            {classification?.results[0]?.actualLabel && <th>Actual Label</th>}
                                        </tr>
                                        </thead>
                                        <tbody>
                                        {paginatedResults?.map((result, index) => {
                                            const isMismatch = result.actualLabel && result.label !== result.actualLabel;
                                            return (
                                                <tr
                                                    key={index}
                                                    onClick={() => navigate(`/datasets/${datasetId}/classifications/${classificationId}/results/${index}`)}
                                                    className={isMismatch ? 'table-danger' : ''}
                                                    style={{cursor: 'pointer'}}
                                                >
                                                    <td className="text-truncate" style={{maxWidth: '300px'}}>
                                                        {result.text}
                                                    </td>
                                                    <td>
                                                        <Badge bg={result.label === 'POSITIVE' ? 'success' : 'danger'}>
                                                            {result.label}
                                                        </Badge>
                                                    </td>
                                                    <td>{renderConfidence(result.score)}</td>
                                                    {result.actualLabel && (
                                                        <td>
                                                            <Badge
                                                                bg={result.actualLabel === 'POSITIVE' ? 'success' : 'danger'}>
                                                                {result.actualLabel}
                                                            </Badge>
                                                        </td>
                                                    )}
                                                </tr>
                                            );
                                        })}
                                        </tbody>
                                    </Table>

                                    {/* Pagination controls remain the same */}
                                    <div className="d-flex justify-content-center">
                                        <Button
                                            variant="outline-primary"
                                            disabled={currentPage === 1}
                                            onClick={() => setCurrentPage(p => p - 1)}
                                        >
                                            Previous
                                        </Button>
                                        <span className="mx-3 my-auto">
                                            Page {currentPage} of {Math.ceil((classification?.results?.length || 0) / itemsPerPage)}
                                        </span>
                                        <Button
                                            variant="outline-primary"
                                            disabled={currentPage * itemsPerPage >= (classification?.results?.length || 0)}
                                            onClick={() => setCurrentPage(p => p + 1)}
                                        >
                                            Next
                                        </Button>
                                    </div>
                                </Card.Body>
                            </Card>
                        </Col>
                    </Row>
                </>
            )}

        </Container><Modal show={showModelModal} onHide={() => setShowModelModal(false)}>
            <Modal.Header closeButton>
                <Modal.Title>Select 3 LLMs</Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <p>Choose 3 models from different providers:</p>
                <div className="model-selection">
                    {availableModels.map((model, index) => (
                        <div key={index} className="mb-2">
                            <input
                                type="checkbox"
                                id={`model-${index}`}
                                checked={selectedModels.includes(`${model.provider}:${model.model}`)}
                                onChange={(e) => {
                                    const modelKey = `${model.provider}:${model.model}`;
                                    let newSelected = [...selectedModels];

                                    if (e.target.checked) {
                                        // Only allow 3 selections
                                        if (newSelected.length < 3) {
                                            newSelected.push(modelKey);
                                        }
                                    } else {
                                        newSelected = newSelected.filter(m => m !== modelKey);
                                    }

                                    setSelectedModels(newSelected);
                                }}
                                disabled={selectedModels.length >= 3 &&
                                    !selectedModels.includes(`${model.provider}:${model.model}`)}/>
                            <label htmlFor={`model-${index}`} className="ms-2">
                                {model.provider} - {model.model}
                            </label>
                        </div>
                    ))}
                </div>
            </Modal.Body>
            <Modal.Footer>
                <Button variant="secondary" onClick={() => setShowModelModal(false)}>
                    Cancel
                </Button>
                <Button
                    variant="primary"
                    onClick={handleSubmitModels}
                    disabled={selectedModels.length !== 3}
                >
                    Submit
                </Button>
            </Modal.Footer>
        </Modal></>
    );
};

export default ClassificationDashboard;