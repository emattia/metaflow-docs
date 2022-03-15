"use strict";(self.webpackChunkdocusaurus=self.webpackChunkdocusaurus||[]).push([[5137],{3905:function(e,t,n){n.d(t,{Zo:function(){return c},kt:function(){return m}});var o=n(7294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);t&&(o=o.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,o)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function l(e,t){if(null==e)return{};var n,o,r=function(e,t){if(null==e)return{};var n,o,r={},a=Object.keys(e);for(o=0;o<a.length;o++)n=a[o],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(o=0;o<a.length;o++)n=a[o],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var s=o.createContext({}),p=function(e){var t=o.useContext(s),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},c=function(e){var t=p(e.components);return o.createElement(s.Provider,{value:t},e.children)},u={inlineCode:"code",wrapper:function(e){var t=e.children;return o.createElement(o.Fragment,{},t)}},d=o.forwardRef((function(e,t){var n=e.components,r=e.mdxType,a=e.originalType,s=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),d=p(n),m=r,f=d["".concat(s,".").concat(m)]||d[m]||u[m]||a;return n?o.createElement(f,i(i({ref:t},c),{},{components:n})):o.createElement(f,i({ref:t},c))}));function m(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var a=n.length,i=new Array(a);i[0]=d;var l={};for(var s in t)hasOwnProperty.call(t,s)&&(l[s]=t[s]);l.originalType=e,l.mdxType="string"==typeof e?e:r,i[1]=l;for(var p=2;p<a;p++)i[p]=n[p];return o.createElement.apply(null,i)}return o.createElement.apply(null,n)}d.displayName="MDXCreateElement"},1410:function(e,t,n){n.r(t),n.d(t,{frontMatter:function(){return s},contentTitle:function(){return p},metadata:function(){return c},toc:function(){return u},default:function(){return f}});var o,r=n(7462),a=n(3366),i=(n(7294),n(3905)),l=["components"],s={},p="Episode 2: Statistics",c={unversionedId:"python/getting-started/tutorials/season-1-the-local-experience/episode02",id:"python/getting-started/tutorials/season-1-the-local-experience/episode02",title:"Episode 2: Statistics",description:"Is this Data Science?",source:"@site/docs/python/getting-started/tutorials/season-1-the-local-experience/episode02.md",sourceDirName:"python/getting-started/tutorials/season-1-the-local-experience",slug:"/python/getting-started/tutorials/season-1-the-local-experience/episode02",permalink:"/metaflow-docs/python/getting-started/tutorials/season-1-the-local-experience/episode02",editUrl:"https://github.dev/obgibson/metaflow-docs/blob/master/docs/python/getting-started/tutorials/season-1-the-local-experience/episode02.md",tags:[],version:"current",frontMatter:{},sidebar:"python",previous:{title:"Episode 1: Playlist",permalink:"/metaflow-docs/python/getting-started/tutorials/season-1-the-local-experience/episode01"},next:{title:"Episode 3: Playlist Redux",permalink:"/metaflow-docs/python/getting-started/tutorials/season-1-the-local-experience/episode03"}},u=[{value:"Is this Data Science?",id:"is-this-data-science",children:[],level:2}],d=(o="TutorialsLink",function(e){return console.warn("Component "+o+" was not imported, exported, or provided by MDXProvider as global scope"),(0,i.kt)("div",e)}),m={toc:u};function f(e){var t=e.components,n=(0,a.Z)(e,l);return(0,i.kt)("wrapper",(0,r.Z)({},m,n,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"episode-2-statistics"},"Episode 2: Statistics"),(0,i.kt)("h2",{id:"is-this-data-science"},"Is this Data Science?"),(0,i.kt)("p",null,"Use metaflow to load the movie metadata CSV file into a Pandas Dataframe and compute some movie genre-specific statistics. These statistics are then used in later examples to improve our playlist generator. You can optionally use the Metaflow client to eyeball the results in a Notebook, and make some simple plots using the Matplotlib library."),(0,i.kt)("p",null,"You can find the tutorial code on ",(0,i.kt)("a",{parentName:"p",href:"https://github.com/Netflix/metaflow/tree/master/metaflow/tutorials/02-statistics"},"GitHub")),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Showcasing:")),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"Fan-out over a set of parameters using Metaflow ",(0,i.kt)("a",{parentName:"li",href:"../../../metaflow/basics#foreach"},"foreach"),"."),(0,i.kt)("li",{parentName:"ul"},"Using external packages like Pandas."),(0,i.kt)("li",{parentName:"ul"},"Plotting results in a Notebook.")),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"Before playing this episode:")),(0,i.kt)("ol",null,(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("inlineCode",{parentName:"li"},"python -m pip install pandas")),(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("inlineCode",{parentName:"li"},"python -m pip install notebook")),(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("inlineCode",{parentName:"li"},"python -m pip install matplotlib"))),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"To play this episode:")),(0,i.kt)("ol",null,(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("inlineCode",{parentName:"li"},"cd metaflow-tutorials")),(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("inlineCode",{parentName:"li"},"python 02-statistics/stats.py show")),(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("inlineCode",{parentName:"li"},"python 02-statistics/stats.py run")),(0,i.kt)("li",{parentName:"ol"},(0,i.kt)("inlineCode",{parentName:"li"},"jupyter-notebook 02-statistics/stats.ipynb"))),(0,i.kt)(d,{link:"../",mdxType:"TutorialsLink"}))}f.isMDXComponent=!0}}]);