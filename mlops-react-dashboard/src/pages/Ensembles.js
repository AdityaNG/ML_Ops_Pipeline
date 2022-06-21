import React from "react";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCheck, faCog, faHome, faSearch } from '@fortawesome/free-solid-svg-icons';
import { Col, Row, Form, Button, ButtonGroup, Breadcrumb, InputGroup, Dropdown } from '@themesberg/react-bootstrap';

import { EnsemblesTable } from "../components/Tables";

import swal from 'sweetalert';
const axios = require('axios');


export default class Ensembles extends React.Component {
	constructor(props) {
		super(props);
		this.state = {
			ensembles: []
		}
		axios.get('http://localhost:5000/ensembles', {}).then((res) => {
			this.setState({
				ensembles: res.data.ensembles
			});
		}).catch((err) => {
			if (err.response && err.response.data && err.response.data.errorMessage) {
				swal({
					text: err.response.data.errorMessage,
					icon: "error",
					type: "error"
			});
		}
		});

	}

	render() {
		return (
			<>
			<div className="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center py-4">
				<div className="d-block mb-4 mb-md-0">
				<Breadcrumb className="d-none d-md-inline-block" listProps={{ className: "breadcrumb-dark breadcrumb-transparent" }}>
					<Breadcrumb.Item><FontAwesomeIcon icon={faHome} /></Breadcrumb.Item>
					<Breadcrumb.Item>ML Ops</Breadcrumb.Item>
					<Breadcrumb.Item active>Ensembles</Breadcrumb.Item>
				</Breadcrumb>
				<h4>Ensembles</h4>
				<p className="mb-0">Your web analytics dashboard template.</p>
				</div>
				<div className="btn-toolbar mb-2 mb-md-0">
				<ButtonGroup>
					<Button variant="outline-primary" size="sm">Share</Button>
					<Button variant="outline-primary" size="sm">Export</Button>
				</ButtonGroup>
				</div>
			</div>

			<div className="table-settings mb-4">
				<Row className="justify-content-between align-items-center">
				<Col xs={8} md={6} lg={3} xl={4}>
					<InputGroup>
					<InputGroup.Text>
						<FontAwesomeIcon icon={faSearch} />
					</InputGroup.Text>
					<Form.Control type="text" placeholder="Search" />
					</InputGroup>
				</Col>
				<Col xs={4} md={2} xl={1} className="ps-md-0 text-end">
					<Dropdown as={ButtonGroup}>
					<Dropdown.Toggle split as={Button} variant="link" className="text-dark m-0 p-0">
						<span className="icon icon-sm icon-gray">
						<FontAwesomeIcon icon={faCog} />
						</span>
					</Dropdown.Toggle>
					<Dropdown.Menu className="dropdown-menu-xs dropdown-menu-right">
						<Dropdown.Item className="fw-bold text-dark">Show</Dropdown.Item>
						<Dropdown.Item className="d-flex fw-bold">
						10 <span className="icon icon-small ms-auto"><FontAwesomeIcon icon={faCheck} /></span>
						</Dropdown.Item>
						<Dropdown.Item className="fw-bold">20</Dropdown.Item>
						<Dropdown.Item className="fw-bold">30</Dropdown.Item>
					</Dropdown.Menu>
					</Dropdown>
				</Col>
				</Row>
			</div>

			<EnsemblesTable ensembles={this.state.ensembles}/>
			</>
		);
	}
};
