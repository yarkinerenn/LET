import { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { Container, Row, Col, Table, Button, Alert, Spinner } from "react-bootstrap";
import axios from "axios";

const DatasetView = () => {
    const { datasetId } = useParams();
    const [dataset, setDataset] = useState<{ filename: string; data: any[] } | null>(null);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchDataset = async () => {
            try {
                const response = await axios.get(`http://localhost:5000/api/dataset/${datasetId}`, {
                    withCredentials: true,
                });
                setDataset(response.data);
            } catch (err) {
                setError("Failed to load dataset.");
            } finally {
                setLoading(false);
            }
        };

        fetchDataset();
    }, [datasetId]);

    // @ts-ignore
    // @ts-ignore
    return (
        <Container className="py-5">
            <Row className="mb-4">
                <Col>
                    <Link to="/datasets">
                        <Button variant="secondary">← Back to Datasets</Button>
                    </Link>
                </Col>
            </Row>

            {loading ? (
                <Spinner animation="border" />
            ) : error ? (
                <Alert variant="danger">{error}</Alert>
            ) : (
                <>
                    <h2 className="text-center mb-4">{dataset?.filename}</h2>
                    {dataset?.data && dataset.data.length > 0 ? (
                        <Table striped bordered hover>
                            <thead>
                            <tr>
                                {dataset?.data?.length > 0 &&
                                    Object.keys(dataset.data[0] || {}).map((key) => (
                                        <th key={key}>{key}</th>
                                    ))}
                            </tr>
                            </thead>
                            <tbody>
                            {dataset?.data?.map((row, index) => (
                                <tr key={index}>
                                    {Object.values(row || {}).map((value, i) => (
                                        <td key={i}>{String(value)}</td>
                                        ))}
                                </tr>
                            ))}
                            </tbody>
                        </Table>
                    ) : (
                        <p className="text-center">No data available.</p>
                    )}
                </>
            )}
        </Container>
    );
};

export default DatasetView;