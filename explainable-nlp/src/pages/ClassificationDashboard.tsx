import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { Container, Row, Col, Card, Table, Alert, Spinner, Button, Badge } from 'react-bootstrap';
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

    return (
        <Container fluid className="py-4">
            <Row className="mb-4">
                <Col>
                    <Link to={`/datasets/${datasetId}`}>
                        <Button variant="outline-secondary">← Back to Dataset</Button>
                    </Link>
                </Col>
            </Row>

            {loading ? (
                <div className="text-center">
                    <Spinner animation="border" />
                </div>
            ) : error ? (
                <Alert variant="danger">{error}</Alert>
            ) : (
                <>
                    <Row className="mb-4">
                        <Col>
                            <h2>Classification Report</h2>
                            <div className="d-flex gap-3 mb-3">
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
                                                    { name: 'Positive', value: stats?.positive || 0 },
                                                    { name: 'Negative', value: stats?.negative || 0 },
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
                                                        fill={index === 0 ? '#4CAF50' : '#F44336'}
                                                    />
                                                ))}
                                            </Pie>
                                            <Tooltip />
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
                                            <CartesianGrid strokeDasharray="3 3" />
                                            <XAxis dataKey="name" />
                                            <YAxis />
                                            <Tooltip />
                                            <RechartsLegend />
                                            <Bar dataKey="value" fill="#8884d8" />
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
                                            {classification?.results[0]?.actualLabel && (
                                                <th>Actual Label</th>
                                            )}
                                        </tr>
                                        </thead>
                                        <tbody>
                                        {paginatedResults?.map((result, index) => (
                                            <tr key={index}>
                                                <td className="text-truncate" style={{ maxWidth: '300px' }}>
                                                    {result.text}
                                                </td>
                                                <td>
                                                    <Badge
                                                        bg={result.label === 'POSITIVE' ? 'success' : 'danger'}
                                                    >
                                                        {result.label}
                                                    </Badge>
                                                </td>
                                                <td>
                                                    {renderConfidence(result.score)}
                                                </td>
                                                {result.actualLabel && (
                                                    <td>
                                                        <Badge
                                                            bg={result.actualLabel === 'POSITIVE' ? 'success' : 'danger'}
                                                        >
                                                            {result.actualLabel}
                                                        </Badge>
                                                    </td>
                                                )}
                                            </tr>
                                        ))}
                                        </tbody>
                                    </Table>

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
        </Container>
    );
};

export default ClassificationDashboard;