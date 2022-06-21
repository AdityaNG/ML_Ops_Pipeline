import React from "react";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBoxOpen, faCartArrowDown, faChartPie, faChevronDown, faClipboard, faCommentDots, faFileAlt, faPlus, faRocket, faSave, faStore } from '@fortawesome/free-solid-svg-icons';
import { Col, Row, Button, Dropdown } from '@themesberg/react-bootstrap';
import { ChoosePhotoWidget, ProfileCardWidget } from "../components/Widgets";
import { GeneralInfoForm } from "../components/Forms";

import Editor from 'react-simple-code-editor';
import { highlight, languages } from 'prismjs/components/prism-core';
import 'prismjs/components/prism-clike';
import 'prismjs/components/prism-javascript';
import 'prismjs/themes/prism.css'; //Example style, you can use another


export default () => {
	const [code, setCode] = React.useState(
		`import numpy as np\nimport matplotlib.pyplot as plt\n\n# Create a random array of size 10x10\nx = np.random.rand(10, 10)\n\n# Plot the array\nplt.imshow(x)\nplt.show()`
	  );	
	
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
			value={code}
			//onValueChange={code => setCode(code)}
			highlight={code => highlight(code, languages.js)}
			padding={10}
			style={{
				fontFamily: '"Fira code", "Fira Mono", monospace',
				fontSize: 12,
			}}
			/>
        </Col>
      </Row>
    </>
  );
};
