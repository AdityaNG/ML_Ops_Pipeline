import React from "react";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBoxOpen, faCartArrowDown, faChartPie, faChevronDown, faClipboard, faCommentDots, faFileAlt, faPlus, faRocket, faSave, faStore } from '@fortawesome/free-solid-svg-icons';
import { Col, Row, Button, Dropdown } from '@themesberg/react-bootstrap';
import { ChoosePhotoWidget, ProfileCardWidget } from "../components/Widgets";
import { GeneralInfoForm } from "../components/Forms";

import Editor from 'react-simple-code-editor';
//import ReactPrismEditor from "react-prism-editor";

import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-clike';
import 'prismjs/components/prism-javascript';
import 'prismjs/components/prism-python';
import 'prismjs/themes/prism.css'; //Example style, you can use another

const hightlightWithLineNumbers = (input, language) =>
  highlight(input, language)
    .split("\n")
    .map((line, i) => `<span class='editorLineNumber'>${i + 1}</span>${line}`)
    .join("\n");

export default class Settings extends React.Component {
	constructor(props) {
		super(props);
		this.state = {
			code: "import numpy as np\nimport matplotlib.pyplot as plt\n\n# Create a random array of size 10x10\nx = np.random.rand(10, 10)\n\n# Plot the array\nplt.imshow(x)\nplt.show()"
		}
		this.setCode = this.setCode.bind(this);
	}

	setCode(new_code) {
		this.setState({
			code: new_code
		})
	}	

	render() {
		// const [code, setCode] = React.useState(
		// 	`import numpy as np\nimport matplotlib.pyplot as plt\n\n# Create a random array of size 10x10\nx = np.random.rand(10, 10)\n\n# Plot the array\nplt.imshow(x)\nplt.show()`
		// );
		const init_code = this.state.code
	
		return (
			<>
			<div className="d-flex justify-content-between flex-wrap flex-md-nowrap align-items-center py-4">
				<Button as={Button} variant="secondary" className="text-dark me-2">
					<FontAwesomeIcon icon={faSave} className="me-2" />
					<span>Save</span>
				</Button>
			</div>

			<Row>
				<Col xs={12} xl={8}>
				<Editor
					value={this.state.code}
					//onValueChange={code => this.setCode(code)}
					onValueChange={this.setCode}
					//highlight={code => hightlightWithLineNumbers(code, languages.py)}
					highlight={code => highlight(code, languages.py)}
					padding={10}
					style={{
						fontFamily: '"Fira code", "Fira Mono", monospace',
						fontSize: 12,
					}}
					/>
				{/*
				<ReactPrismEditor
					language={'python'}
					code={this.state.code}
					lineNumber={lineNumber}
					readOnly={readOnly}
					clipboard={true}
					changeCode={code => {
						this.code = code
						console.log(code)
					}}
					/>
				*/}
				</Col>
			</Row>
			</>
		);
	}
};
