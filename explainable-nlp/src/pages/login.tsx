import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Form, Button, Alert, Container, Row, Col } from 'react-bootstrap';
import { useAuth } from '../modules/auth';

export function Login() {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        try {
            const response = await fetch('http://localhost:5000/api/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password }),
                credentials: 'include',
            });

            if (response.ok) {
                const data = await response.json();
                login(data.user);
                navigate('/');
            } else {
                const errorData = await response.json();
                setError(errorData.error);
            }
        } catch (err) {
            setError('An error occurred during login');
        }
    };

    return (
        <Container className="py-5">
            <Row className="justify-content-center">
                <Col md={6} lg={4}>
                    <div className="auth-card">
                        <h2 className="text-center mb-4">Welcome Back</h2>
                        {error && <Alert variant="danger">{error}</Alert>}
                        <Form onSubmit={handleSubmit}>
                            <Form.Group className="mb-3">
                                <Form.Label>Email address</Form.Label>
                                <Form.Control
                                    type="email"
                                    placeholder="Enter email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    required
                                />
                            </Form.Group>

                            <Form.Group className="mb-4">
                                <Form.Label>Password</Form.Label>
                                <Form.Control
                                    type="password"
                                    placeholder="Password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                />
                            </Form.Group>

                            <Button variant="primary" className="w-100 mb-3" type="submit">
                                Sign In
                            </Button>

                            <div className="text-center">
                                <small>
                                    Don't have an account?{' '}
                                    <a href="/register" className="text-primary">
                                        Register here
                                    </a>
                                </small>
                            </div>
                        </Form>
                    </div>
                </Col>
            </Row>
        </Container>
    );
}