import React from "react";
import { Container, Row, Col, Card, Button, Image } from "react-bootstrap";
import { useNavigate } from "react-router-dom";
import heroImage from "../assets/ai.png"; // Resmi buraya koy

function HomePage() {
    const navigate = useNavigate();

    return (
        <Container className="mt-5">
            <Row className="justify-content-center mb-4">
                <Col md={8} className="text-center">
                    <h1 className="display-4 p-2">LET: LLM Explanation Tool</h1>
                    <Image src={heroImage} fluid rounded className="mb-4 shadow"/>

                    <p className="lead">
                    “Explore the faithfulness and plausibility of AI explanations with LET. Upload datasets or test single entries, run models from any major provider or BERT locally, and compare self-explanations, post-hoc rationales, and SHAP-augmented insights. Designed for research and user studies, LET makes explanation quality measurable and transparent.”

                    </p>
                </Col>
            </Row>
            <Row className="justify-content-center">
                <Col md={5}>
                    <Card className="shadow mb-4">
                        <Card.Body className="text-center">
                            <Card.Title>Dashboard</Card.Title>
                            <Card.Text>
                                Dive into sentiment classification and real-time analysis.
                            </Card.Text>
                            <Button variant="dark" onClick={() => navigate("/dashboard")}>
                                Go to Dashboard
                            </Button>
                        </Card.Body>
                    </Card>
                </Col>
                <Col md={5}>
                    <Card className="shadow mb-4">
                        <Card.Body className="text-center">
                            <Card.Title>Datasets</Card.Title>
                            <Card.Text>
                                Manage and classify your datasets with ease.
                            </Card.Text>
                            <Button variant="dark" onClick={() => navigate("/datasets")}>
                                Go to Datasets
                            </Button>
                        </Card.Body>
                    </Card>
                </Col>
            </Row>
        </Container>
    );
}

export default HomePage;